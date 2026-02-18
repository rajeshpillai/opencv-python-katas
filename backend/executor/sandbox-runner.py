"""
sandbox-runner.py — Isolated execution script.

This script is run as a subprocess by sandbox.py.
It intercepts cv2.imshow() and saves the image to a temp file,
then prints the base64-encoded PNG to stdout.

ONLY cv2 and numpy imports are allowed.
"""

import sys
import os
import ast
import base64
import tempfile
import builtins

# ── Safety: block dangerous builtins ──────────────────────────────────────────

_BLOCKED_BUILTINS = {"open", "exec", "eval", "compile", "__import__"}
_original_import = builtins.__import__

ALLOWED_MODULES = {"cv2", "numpy", "np", "math", "random"}


def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in ALLOWED_MODULES:
        raise ImportError(
            f"Import '{name}' is not allowed. "
            f"Only cv2 and numpy are permitted."
        )
    return _original_import(name, *args, **kwargs)


builtins.__import__ = _safe_import

# ── Intercept cv2.imshow ──────────────────────────────────────────────────────

import cv2
import numpy as np

_output_images: list = []
_output_path = tempfile.mktemp(suffix=".png")

_original_imshow = cv2.imshow


def _capture_imshow(winname, mat):
    """Replace cv2.imshow with image capture."""
    _output_images.append(mat.copy())


cv2.imshow = _capture_imshow
cv2.waitKey = lambda _=0: 0
cv2.destroyAllWindows = lambda: None
cv2.destroyWindow = lambda _: None

# ── Run user code ─────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("ERROR: No code file provided", file=sys.stderr)
    sys.exit(1)

code_file = sys.argv[1]
with open(code_file, "r") as f:
    user_code = f.read()

try:
    exec(compile(user_code, "<kata>", "exec"), {"cv2": cv2, "np": np, "numpy": np})
except Exception as e:
    print(f"EXEC_ERROR:{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)

# ── Output result ─────────────────────────────────────────────────────────────

if _output_images:
    img = _output_images[-1]
    # Ensure it's a valid image array
    if img is not None and img.size > 0:
        success, buf = cv2.imencode(".png", img)
        if success:
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            print(f"IMAGE:{b64}")
        else:
            print("ERROR: Failed to encode image", file=sys.stderr)
    else:
        print("ERROR: Empty image", file=sys.stderr)
else:
    # No imshow called — check if user created an image variable
    print("INFO: No cv2.imshow() called. Call cv2.imshow('result', your_image) to display output.")
