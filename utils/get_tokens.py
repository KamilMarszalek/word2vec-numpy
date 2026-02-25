import sys
from pathlib import Path

if __name__ == "__main__":
    in_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])
    n_tokens = int(sys.argv[3])

    text = in_file.read_text(encoding="utf-8")
    tokens = text.split()
    subset = " ".join(tokens[:n_tokens])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(subset, encoding="utf-8")
    print(f"Wrote {min(len(tokens), n_tokens)} tokens to {out_file}")
