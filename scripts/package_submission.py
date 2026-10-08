import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_OUT = PROJECT_ROOT / "ClaimLens_Project_Submission.zip"

EXCLUDE_DIRS = {
    "venv", ".venv", ".git", "__pycache__", ".pytest_cache", ".deepeval"
}
EXCLUDE_FILES = {
    ".env", "ClaimLens_Project_Submission.zip"
}
EXCLUDE_EXTS = {
    ".pyc", ".pyo"
}

def package_project():
    print(f"Creating submission package at {ZIP_OUT}...")
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PROJECT_ROOT.rglob("*"):
            if file_path.is_dir():
                continue
            
            # Check exclusions
            parts = file_path.relative_to(PROJECT_ROOT).parts
            if any(part in EXCLUDE_DIRS for part in parts):
                continue
            if file_path.name in EXCLUDE_FILES:
                continue
            if file_path.suffix in EXCLUDE_EXTS:
                continue
            
            rel_path = file_path.relative_to(PROJECT_ROOT)
            zf.write(file_path, arcname=str(rel_path))
            print(f"Added: {rel_path}")

    print(f"\nSuccessfully created {ZIP_OUT} ({ZIP_OUT.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    package_project()
