"""Print a local development-environment report without installing anything."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


MODULES = {
    "numpy": "NumPy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "pytest": "pytest",
    "fastapi": "FastAPI (later API step)",
    "torch": "PyTorch (transformer step; intentionally optional now)",
    "transformers": "Transformers (transformer step; intentionally optional now)",
}


def command_version(command: str, args: list[str]) -> str:
    executable = shutil.which(command)
    if not executable:
        return "not found"
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=5, check=False
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        return output[0] if output else f"found ({executable})"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"found, but version check failed: {exc}"


def main() -> None:
    print("ScamShield India — environment check")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Executable: {sys.executable}")
    print(f"Git: {command_version('git', ['--version'])}")
    print(f"Docker: {command_version('docker', ['--version'])} (optional for Step 1)")
    print("Packages:")
    for module, description in MODULES.items():
        status = "installed" if importlib.util.find_spec(module) else "not installed"
        optional = " [optional later]" if module in {"fastapi", "torch", "transformers"} else ""
        print(f"  - {description}: {status}{optional}")

    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 11:
        print("Python check: PASS (3.11+ detected).")
    else:
        print("Python check: REVIEW (Python 3.11 or newer is recommended).")

    missing_core = [name for name in ("numpy", "pandas", "sklearn", "pytest")
                    if importlib.util.find_spec(name) is None]
    if missing_core:
        print("Core packages still needed: " + ", ".join(missing_core))
    else:
        print("Core package check: PASS.")
    print("Note: this script only reports your local machine; it does not install packages.")


if __name__ == "__main__":
    main()
