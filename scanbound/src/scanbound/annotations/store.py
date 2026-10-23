"""
The annotation store: the single source of truth for every annotation of a project.

Pure Python, no Qt.  The GUI's viewers *render* the store and send edits to it; scripts and tests
use exactly the same calls.  Every change is a command that can be undone, and listeners get one
:class:`StoreEvent` per command (or per :meth:`AnnotationStore.transaction`), which is how views
stay in sync without polling.

    store = AnnotationStore()
    a = store.add(Annotation("scan.sxm::Z", "feature", Rect(10, 10, 5, 5), label="molecule"))
    with store.transaction("Relabel and accept"):
        store.set_label([a.id], "9AP (S)")
        store.accept([a.id])
    store.undo()                       # both changes are reverted together
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Union

from .annotation import STATUSES, Annotation
from .kinds import Issue, ValidationContext, validate

Ids = Union[str, Iterable[str]]


def _ids(ids: Ids) -> list[str]:
    """One id or several (a bare string is one id, not its characters)."""
    return [ids] if isinstance(ids, str) else list(ids)


Listener = Callable[["StoreEvent"], None]


@dataclass(frozen=True)
class Change:
    before: Optional[Annotation]
    after: Optional[Annotation]

    @property
    def action(self) -> str:
        if self.before is None:
            return "added"
        if self.after is None:
            return "removed"
        return "updated"

    @property
    def id(self) -> str:
        return (self.after or self.before).id  # type: ignore[union-attr]

    @property
    def image_id(self) -> str:
        return (self.after or self.before).image_id  # type: ignore[union-attr]

    def inverse(self) -> "Change":
        return Change(self.after, self.before)


@dataclass(frozen=True)
class StoreEvent:
    changes: tuple[Change, ...]
    description: str = ""
    source: str = "edit"          # "edit" / "undo" / "redo" / "load"

    def ids(self, action: Optional[str] = None) -> list[str]:
        return [c.id for c in self.changes if action is None or c.action == action]

    @property
    def image_ids(self) -> set[str]:
        out = set()
        for c in self.changes:
            for a in (c.before, c.after):
                if a is not None:
                    out.add(a.image_id)
        return out


@dataclass
class _Command:
    description: str
    changes: list[Change] = field(default_factory=list)


class AnnotationStore:
    def __init__(self, annotations: Iterable[Annotation] = (), max_undo: int = 500):
        self._items: dict[str, Annotation] = {}
        self._by_image: dict[str, dict[str, None]] = {}
        self._listeners: list[Listener] = []
        self._undo: list[_Command] = []
        self._redo: list[_Command] = []
        self._tx: Optional[_Command] = None
        self._tx_depth = 0
        self.max_undo = max_undo
        self.revision = 0
        self.auto_accept = False       # add proposals as accepted (the app does this while review is off)
        for a in annotations:
            self._apply(Change(None, a))

    # =========================================================================== reading
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(list(self._items.values()))

    def __contains__(self, ann_id: object) -> bool:
        return ann_id in self._items

    def get(self, ann_id: str) -> Optional[Annotation]:
        return self._items.get(ann_id)

    def __getitem__(self, ann_id: str) -> Annotation:
        return self._items[ann_id]

    def image_ids(self) -> list[str]:
        return [k for k, v in self._by_image.items() if v]

    def for_image(self, image_id: str, kinds: Optional[Sequence[str]] = None,
                  statuses: Optional[Sequence[str]] = None, labels: Optional[Sequence[Optional[str]]] = None
                  ) -> list[Annotation]:
        out = []
        for i in self._by_image.get(image_id, ()):
            a = self._items[i]
            if kinds is not None and a.kind not in kinds:
                continue
            if statuses is not None and a.status not in statuses:
                continue
            if labels is not None and a.label not in labels:
                continue
            out.append(a)
        return out

    def query(self, predicate: Callable[[Annotation], bool]) -> list[Annotation]:
        return [a for a in self._items.values() if predicate(a)]

    def children(self, ann_id: str) -> list[Annotation]:
        parent = self._items.get(ann_id)
        if parent is None:
            return []
        return [a for a in self.for_image(parent.image_id) if a.parent == ann_id]

    def labels(self) -> list[str]:
        """Labels in use, in order of first appearance."""
        return list(dict.fromkeys(a.label for a in self._items.values() if a.label))

    def counts(self, image_id: Optional[str] = None, by: str = "label") -> Counter:
        items = self.for_image(image_id) if image_id is not None else self._items.values()
        return Counter(getattr(a, by) for a in items)

    def hit_test(self, image_id: str, x: float, y: float, tolerance: float = 3.0,
                 kinds: Optional[Sequence[str]] = None) -> list[Annotation]:
        """Annotations under (x, y), best first: nearest outline, then smallest area (so a small
        feature inside a big one, or one of two touching features, is picked precisely)."""
        hits = []
        for a in self.for_image(image_id, kinds=kinds):
            if a.status == "rejected":
                continue
            d = a.geometry.distance_to(x, y)
            if d <= tolerance:
                hits.append((d, a.geometry.area(), a))
        hits.sort(key=lambda h: (h[0], h[1]))
        return [h[2] for h in hits]

    def issues(self, image_id: Optional[str] = None, image_shape: Optional[tuple[int, int]] = None) -> list[Issue]:
        ctx = ValidationContext(image_shape, self)
        items = self.for_image(image_id) if image_id is not None else list(self._items.values())
        return [i for a in items for i in validate(a, ctx)]

    # =========================================================================== events
    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def _emit(self, changes: Sequence[Change], description: str, source: str) -> None:
        if not changes:
            return
        self.revision += 1
        ev = StoreEvent(tuple(changes), description, source)
        for fn in list(self._listeners):
            fn(ev)

    # =========================================================================== writing
    def _apply(self, ch: Change) -> None:
        if ch.before is not None:
            self._items.pop(ch.before.id, None)
            self._by_image.get(ch.before.image_id, {}).pop(ch.before.id, None)
        if ch.after is not None:
            self._items[ch.after.id] = ch.after
            self._by_image.setdefault(ch.after.image_id, {})[ch.after.id] = None

    def _commit(self, changes: list[Change], description: str) -> None:
        changes = [c for c in changes if c.before is not c.after]
        if not changes:
            return
        for c in changes:
            self._apply(c)
        if self._tx is not None:
            self._tx.changes.extend(changes)
            return
        self._push(_Command(description, changes))
        self._emit(changes, description, "edit")

    def _push(self, cmd: _Command) -> None:
        self._undo.append(cmd)
        del self._undo[: -self.max_undo]
        self._redo.clear()

    @contextmanager
    def transaction(self, description: str = "Edit") -> Iterator[None]:
        """Group several edits into one undo step and one event."""
        if self._tx is not None:                    # nested: join the outer one
            self._tx_depth += 1
            try:
                yield
            finally:
                self._tx_depth -= 1
            return
        self._tx = _Command(description)
        try:
            yield
        except BaseException:
            for c in reversed(self._tx.changes):    # roll back on error
                self._apply(c.inverse())
            self._tx = None
            raise
        cmd, self._tx = self._tx, None
        if cmd.changes:
            self._push(cmd)
            self._emit(cmd.changes, description, "edit")

    def add(self, annotations: Union[Annotation, Iterable[Annotation]], description: str = "") -> list[Annotation]:
        items = [annotations] if isinstance(annotations, Annotation) else list(annotations)
        if self.auto_accept:
            items = [replace(a, status="accepted") if a.status == "proposed" else a for a in items]
        for a in items:
            if a.id in self._items:
                raise KeyError(f"annotation {a.id} already exists")
        self._commit([Change(None, a) for a in items], description or f"Add {len(items)} annotation(s)")
        return items

    def remove(self, ids: Ids, cascade: bool = True, description: str = "") -> list[Annotation]:
        """Remove annotations (and, with ``cascade``, the ones that belong to them)."""
        ids = _ids(ids)
        todo = list(dict.fromkeys(i for i in ids if i in self._items))
        if cascade:
            k = 0
            while k < len(todo):
                todo.extend(c.id for c in self.children(todo[k]) if c.id not in todo)
                k += 1
        gone = [self._items[i] for i in todo]
        self._commit([Change(a, None) for a in gone], description or f"Delete {len(gone)} annotation(s)")
        return gone

    def update(self, ann_id: str, description: str = "", **changes: Any) -> Annotation:
        old = self._items[ann_id]
        new = old.evolve(**changes)
        self._commit([Change(old, new)], description or "Edit annotation")
        return new

    def update_many(self, ids: Ids, description: str = "", **changes: Any) -> list[Annotation]:
        ids = _ids(ids)
        olds = [self._items[i] for i in dict.fromkeys(ids) if i in self._items]
        pairs = [Change(o, o.evolve(**changes)) for o in olds]
        self._commit(pairs, description or f"Edit {len(pairs)} annotation(s)")
        return [p.after for p in pairs]  # type: ignore[misc]

    def replace(self, ann: Annotation, description: str = "") -> Annotation:
        old = self._items[ann.id]
        self._commit([Change(old, ann)], description or "Edit annotation")
        return ann

    def set_label(self, ids: Ids, label: Optional[str]) -> list[Annotation]:
        ids = _ids(ids)
        ids = [i for i in ids if i in self._items and self._items[i].label != label]
        return self.update_many(ids, f"Label {len(ids)} as {label!r}", label=label)

    def set_status(self, ids: Ids, status: str) -> list[Annotation]:
        ids = _ids(ids)
        if status not in STATUSES:
            raise ValueError(status)
        ids = [i for i in ids if i in self._items and self._items[i].status != status]
        return self.update_many(ids, f"Mark {len(ids)} as {status}", status=status)

    def accept(self, ids: Ids) -> list[Annotation]:
        """Proposed -> accepted (manual ones stay manual)."""
        ids = _ids(ids)
        return self.set_status([i for i in ids if i in self._items and self._items[i].status in ("proposed", "rejected")],
                               "accepted")

    def reject(self, ids: Ids) -> list[Annotation]:
        ids = _ids(ids)
        return self.set_status([i for i in ids if i in self._items and self._items[i].status != "manual"], "rejected")

    def move(self, ids: Ids, dx: float, dy: float) -> list[Annotation]:
        ids = _ids(ids)
        olds = [self._items[i] for i in ids if i in self._items]
        pairs = [Change(o, o.evolve(geometry=o.geometry.translated(dx, dy))) for o in olds]
        self._commit(pairs, f"Move {len(pairs)} annotation(s)")
        return [p.after for p in pairs]  # type: ignore[misc]

    def clear_image(self, image_id: str, statuses: Optional[Sequence[str]] = None) -> list[Annotation]:
        ids = [a.id for a in self.for_image(image_id, statuses=statuses)]
        return self.remove(ids, cascade=False, description=f"Clear {len(ids)} annotation(s)")

    def rename_label(self, old: str, new: Optional[str]) -> list[Annotation]:
        ids = [a.id for a in self._items.values() if a.label == old]
        return self.update_many(ids, f"Rename label {old!r} to {new!r}", label=new)

    # =========================================================================== undo
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_text(self) -> str:
        return self._undo[-1].description if self._undo else ""

    def redo_text(self) -> str:
        return self._redo[-1].description if self._redo else ""

    def undo(self) -> bool:
        if not self._undo:
            return False
        cmd = self._undo.pop()
        inv = [c.inverse() for c in reversed(cmd.changes)]
        for c in inv:
            self._apply(c)
        self._redo.append(cmd)
        self._emit(inv, cmd.description, "undo")
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cmd = self._redo.pop()
        for c in cmd.changes:
            self._apply(c)
        self._undo.append(cmd)
        self._emit(cmd.changes, cmd.description, "redo")
        return True

    def clear_history(self) -> None:
        self._undo.clear()
        self._redo.clear()

    # =========================================================================== serialisation
    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._items.values()]

    def load(self, items: Iterable[Union[Annotation, dict]], description: str = "Load annotations") -> None:
        """Replace everything (not undoable; listeners get one 'load' event)."""
        old = list(self._items.values())
        self._items.clear()
        self._by_image.clear()
        new = [a if isinstance(a, Annotation) else Annotation.from_dict(a) for a in items]
        for a in new:
            self._apply(Change(None, a))
        self.clear_history()
        self._emit([Change(a, None) for a in old] + [Change(None, a) for a in new], description, "load")
