"""Alias d'import de la cinématique : la logique a été déplacée dans le paquet robot_arm/.

Garde `python ik.py ...` fonctionnel et les imports existants (`import ik`)
valides. Préférer `import robot_arm` (ou `from robot_arm import ...`) dans le
code nouveau ; le parseur de JSON de séquences est robot_arm.sequence, la
cinématique robot_arm.kinematics, la ligne de commande main.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_arm import *  # noqa: E402,F403
from robot_arm import __all__  # noqa: E402
from robot_arm.calibration import _mid  # noqa: E402,F401
from main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
