"""Import bridge to Stage III's `iste` package, which lives in the sibling `project-v5`
repository rather than inside this one.

CLAUDE.md: "Reuse compatible Stage III components for interpolation, geometry/Earth
conversion, environment access, and boundary concepts... Do not modify Stage III
behavior as a side effect of forward-propagation work." This module makes `iste`
importable without copying or editing any of its files: it adds the `project-v5` repo
root to `sys.path` so `import iste...` resolves to the real, unmodified Stage III
source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VAR = "ISTE_STAGE3_PATH"


def stage3_root() -> Path:
    """Locate the `project-v5` repo root (the directory containing `iste/`).

    Resolution order: the `ISTE_STAGE3_PATH` environment variable if set, otherwise the
    sibling `project-v5` directory next to this repo
    (`forward_prop_final/../project-v5`) -- how the two repos are actually laid out
    under `SIH-2026/` (verified in the F00 audit).
    """
    override = os.environ.get(_ENV_VAR)
    root = Path(override).resolve() if override else Path(__file__).resolve().parents[2] / "project-v5"
    if not (root / "iste" / "__init__.py").exists():
        raise FileNotFoundError(
            f"Stage III package not found at {root / 'iste'}. Set the {_ENV_VAR} "
            "environment variable to the project-v5 repo root if it is not a sibling "
            "of this repo."
        )
    return root


def ensure_iste_importable() -> None:
    """Add the Stage III repo root to `sys.path` (idempotent) so `import iste...` works.

    Safe to call repeatedly (e.g. once per module that needs `iste`); it does not
    re-import or reload anything, only ensures the path is present.
    """
    root = str(stage3_root())
    if root not in sys.path:
        sys.path.insert(0, root)
