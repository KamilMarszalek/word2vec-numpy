import sys
import zipfile
from pathlib import Path

if __name__ == "__main__":
    zip_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    expected = out_dir / "text8"
    if not expected.exists():
        raise SystemExit("Extraction finished, but 'text8' file was not found.")
    print(f"Extracted text8 to: {expected}")
