"""Verify that torch is at the documented safe version.

pip-audit cannot audit torch: the package is installed from the
PyTorch CPU index (download.pytorch.org/whl/cpu), which is not on
PyPI. See docs/security/torch-risk-assessment.md.

If this check fails, it means torch was bumped without updating the
security documentation. Either revert the bump or re-audit manually
and update EXPECTED_VERSION_PREFIX below.
"""

from __future__ import annotations

import sys

import torch

# Единственная версия torch, для которой проведён security-аудит.
# При обновлении — перепроверить вручную и обновить
# docs/security/torch-risk-assessment.md.
EXPECTED_VERSION_PREFIX = "2.5.0"
EXPECTED_SUFFIX = "+cpu"


def main() -> int:
    installed = torch.__version__
    version_part = installed.split("+")[0]

    print(f"torch installed: {installed}")
    print(f"torch expected:  {EXPECTED_VERSION_PREFIX}*")

    if not version_part.startswith(EXPECTED_VERSION_PREFIX):
        print(
            f"ERROR: torch {installed} does not match expected "
            f"{EXPECTED_VERSION_PREFIX}*",
            file=sys.stderr,
        )
        print(
            "pip-audit does not cover torch (CPU index). "
            "If the bump is intentional, re-audit torch manually and "
            "update docs/security/torch-risk-assessment.md, then update "
            "EXPECTED_VERSION_PREFIX in .github/scripts/verify_torch.py.",
            file=sys.stderr,
        )
        return 1

    if not installed.endswith(EXPECTED_SUFFIX):
        print(
            f"ERROR: torch {installed} is not a CPU-only build "
            f"(expected suffix {EXPECTED_SUFFIX})",
            file=sys.stderr,
        )
        print(
            "This project requires CPU-only torch from "
            "download.pytorch.org/whl/cpu. "
            "See docs/security/torch-risk-assessment.md.",
            file=sys.stderr,
        )
        return 1

    print("Torch is not covered by pip-audit.")
    print(
        f"Separate Torch assessment: passed "
        f"({installed}, CPU-only)."
    )
    print("See docs/security/torch-risk-assessment.md for details.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
