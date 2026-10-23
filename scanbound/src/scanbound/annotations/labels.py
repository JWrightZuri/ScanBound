"""
The label set of a project: names, colours and number keys, without Qt.

Labels are the *classes* of annotations ("9AP (S)", "GNR", "vacancy").  The order is the order
of the number keys (1..9, 0).  Colours follow the old viewer's palette so existing projects look
the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

# the old viewer's ROI_COLORS without its first entry (which was the "Empty Label" grey)
PALETTE = ["#D81B60", "#1E88E5", "#D49713", "#4CDBB0", "#D55E00", "#CC79A7", "#F0E442",
           "#56B4E9", "#009E73", "#E69F00", "#0072B2", "#F5E400", "#FFFFFF", "#000000"]
UNLABELED_COLOR = "#B9B1B4"


@dataclass
class LabelSet:
    names: list[str] = field(default_factory=list)
    colors: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = list(dict.fromkeys(n for n in self.names if n))

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def color(self, name: Optional[str]) -> str:
        if not name:
            return UNLABELED_COLOR
        if name in self.colors:
            return self.colors[name]
        if name in self.names:
            return PALETTE[self.names.index(name) % len(PALETTE)]
        return PALETTE[sum(map(ord, name)) % len(PALETTE)]

    def key_for(self, name: str) -> Optional[int]:
        """Number key (1..9, then 0) of a label, None past the tenth."""
        if name not in self.names:
            return None
        i = self.names.index(name)
        return (i + 1) % 10 if i < 10 else None

    def by_key(self, key: int) -> Optional[str]:
        i = 9 if key == 0 else key - 1
        return self.names[i] if 0 <= i < len(self.names) else None

    def add(self, name: str, color: Optional[str] = None) -> bool:
        name = name.strip()
        if not name or name in self.names:
            return False
        self.names.append(name)
        if color:
            self.colors[name] = color
        return True

    def merge(self, names: Iterable[Optional[str]]) -> list[str]:
        return [n for n in names if n and self.add(n)]

    def rename(self, old: str, new: str) -> None:
        if old not in self.names or not new or new in self.names:
            raise ValueError(f"cannot rename {old!r} to {new!r}")
        self.names[self.names.index(old)] = new
        if old in self.colors:
            self.colors[new] = self.colors.pop(old)

    def remove(self, name: str) -> None:
        if name in self.names:
            self.names.remove(name)
        self.colors.pop(name, None)

    def move(self, name: str, index: int) -> None:
        self.names.remove(name)
        self.names.insert(max(0, min(index, len(self.names))), name)

    def to_dict(self) -> dict:
        return {"names": list(self.names), "colors": dict(self.colors)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LabelSet":
        d = d or {}
        return cls(list(d.get("names") or []), dict(d.get("colors") or {}))
