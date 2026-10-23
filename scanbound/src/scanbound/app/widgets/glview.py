"""A pyqtgraph GLViewWidget that is safe to use in several windows at once."""

from __future__ import annotations

import os
import sys

_GL_VIEW = None
_SHADER_OWNER: dict[type, int] = {}               # pyqtgraph item class -> GL share group its shader belongs to


def gl_view_class():  # noqa: ANN201
    """GLViewWidget that recompiles pyqtgraph's shaders when they belong to another GL context.

    pyqtgraph keeps one compiled shader per item class.  Without shared contexts (the application
    attribute AA_ShareOpenGLContexts, set in ``mainwindow.run`` before the QApplication exists) a second
    3D view, or this one after Qt recreates its context, would use a program id that is invalid there
    (GLError 1281 in glUseProgram).
    """
    global _GL_VIEW
    if _GL_VIEW is not None:
        return _GL_VIEW
    import pyqtgraph.opengl as gl
    from PyQt6 import sip
    from PyQt6.QtGui import QOpenGLContext

    item_classes = [c for c in (getattr(gl, n, None) for n in ("GLScatterPlotItem", "GLLinePlotItem", "GLMeshItem",
                                                                 "GLImageItem", "GLSurfacePlotItem"))
                    if c is not None and "_shaderProgram" in c.__dict__]

    class View(gl.GLViewWidget):
        def paintGL(self, *args, **kw):  # noqa: ANN002, ANN003, ANN202
            ctx = QOpenGLContext.currentContext()
            group = sip.unwrapinstance(ctx.shareGroup()) if ctx is not None else 0
            for c in item_classes:
                if c.__dict__.get("_shaderProgram") is not None and _SHADER_OWNER.get(c) != group:
                    c._shaderProgram = None
            try:
                return super().paintGL(*args, **kw)
            finally:
                for c in item_classes:
                    if c.__dict__.get("_shaderProgram") is not None:
                        _SHADER_OWNER[c] = group

    _GL_VIEW = View
    return View


GL_ENV = "SCANBOUND_OPENGL"


def configure_opengl() -> str:
    """Choose the OpenGL context for the 3D views; call before the QApplication exists.

    By default each platform keeps Qt's own choice, which pyqtgraph supports everywhere: a compatibility
    context on Windows and Linux, OpenGL 2.1 on macOS.  ``SCANBOUND_OPENGL`` overrides it when a driver
    misbehaves:

        core      OpenGL 4.1 core profile (the newest macOS offers; pyqtgraph runs its shaders in ES2 mode)
        legacy    OpenGL 2.1
        software  a software renderer (Windows: Qt's bundled one; Linux: Mesa llvmpipe; ignored on macOS)

    Returns the mode in effect ("default" when nothing was changed).
    """
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtGui import QSurfaceFormat

    mode = os.environ.get(GL_ENV, "").strip().lower()
    if mode == "core":
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)
    elif mode == "legacy":
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
        fmt.setDepthBufferSize(24)
        QSurfaceFormat.setDefaultFormat(fmt)
    elif mode == "software":
        if sys.platform == "win32":
            QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
        elif sys.platform.startswith("linux"):
            os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    else:
        mode = "default"
    return mode
