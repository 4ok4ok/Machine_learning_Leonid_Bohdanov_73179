"""Print Python, platform and every pinned package version; save to results/versions.txt."""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def pinned_packages() -> list[str]:
    req = ROOT / "requirements.txt"
    return [line.split("==")[0].strip() for line in req.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def main() -> None:
    lines = [
        f"python=={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor() or 'unknown'}",
        f"executable={sys.executable}",
        "",
    ]
    for pkg in pinned_packages():
        try:
            lines.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            lines.append(f"{pkg}==NOT INSTALLED")
    text = "\n".join(lines)
    print(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "versions.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
