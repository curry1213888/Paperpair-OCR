"""Install dev dependencies and git hooks for consistent formatting with CI."""

from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    run([sys.executable, "-m", "pre_commit", "install"])
    run([sys.executable, "-m", "pre_commit", "install", "--hook-type", "pre-push"])
    print("\nDev setup complete.")
    print("Hooks installed: commit + push will run the same checks as GitHub Actions.")
    print("Manual check: pre-commit run --all-files")


if __name__ == "__main__":
    main()
