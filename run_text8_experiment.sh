#!/usr/bin/env bash
set -euo pipefail

DATASET="data/text8"
OUT_ROOT="experiments/text8"
SEED=42

DOWNLOAD_TEXT8=0
FORCE_DOWNLOAD_TEXT8=0
TEXT8_URL="http://mattmahoney.net/dc/text8.zip"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --download-text8         Download text8 automatically if missing
  --force-download-text8   Re-download text8 even if data/text8 exists
  --dataset PATH           Path to dataset file (default: data/text8)
  --out-root PATH          Output root directory (default: experiments/text8)
  --seed N                 Random seed (default: 42)
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-text8)
      DOWNLOAD_TEXT8=1
      shift
      ;;
    --force-download-text8)
      DOWNLOAD_TEXT8=1
      FORCE_DOWNLOAD_TEXT8=1
      shift
      ;;
    --dataset)
      DATASET="${2:?Missing value for --dataset}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:?Missing value for --out-root}"
      shift 2
      ;;
    --seed)
      SEED="${2:?Missing value for --seed}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done


DATASET="data/text8"
OUT_ROOT="experiments/text8"
SEED=42


if [[ ! -f "main.py" ]]; then
  echo "ERROR: main.py not found. Run this script from the project root." >&2
  exit 1
fi

download_text8_if_needed() {
  local dataset_path="$1"

  
  if [[ "$DOWNLOAD_TEXT8" -ne 1 ]]; then
    return
  fi

  local dataset_dir
  dataset_dir="$(dirname "$dataset_path")"
  local zip_path="${dataset_dir}/text8.zip"

  mkdir -p "$dataset_dir"

  if [[ -f "$dataset_path" && "$FORCE_DOWNLOAD_TEXT8" -ne 1 ]]; then
    echo "text8 already exists: $dataset_path (skipping download)"
    return
  fi

  echo "==> Downloading text8 from: $TEXT8_URL"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$zip_path" "$TEXT8_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$zip_path" "$TEXT8_URL"
  else
    echo "ERROR: Neither curl nor wget is available for downloading text8." >&2
    exit 1
  fi

  echo "==> Extracting $zip_path -> $dataset_dir"

  
  uv run python -m utils.unzip "$zip_path" "$dataset_dir" || {
    echo "ERROR: Failed to unzip $zip_path" >&2
    exit 1
  }
}

download_text8_if_needed "$DATASET"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset file not found: $DATASET" >&2
  echo "Use --download-text8 or put text8 in ./data/text8" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT/subsets" "$OUT_ROOT/models" "$OUT_ROOT/logs" "$OUT_ROOT/reports"

echo "==> Using dataset: $DATASET"
echo "==> Output root: $OUT_ROOT"


create_subset() {
  local in_file="$1"
  local out_file="$2"
  local n_tokens="$3"

  if [[ -f "$out_file" ]]; then
    echo "Subset already exists: $out_file"
    return
  fi

  echo "Creating subset: $out_file (${n_tokens} tokens)"
  uv run python -m utils.get_tokens "$in_file" "$out_file" "$n_tokens"
}

create_subset "$DATASET" "$OUT_ROOT/subsets/text8_100k.txt" 100000
create_subset "$DATASET" "$OUT_ROOT/subsets/text8_500k.txt" 500000
create_subset "$DATASET" "$OUT_ROOT/subsets/text8_1m.txt"   1000000


train_model() {
  local name="$1"
  local dataset="$2"
  local window_size="$3"
  local emb_dim="$4"
  local epochs="$5"
  local lr="$6"
  local neg="$7"

  local model_dir="$OUT_ROOT/models/$name"
  local log_file="$OUT_ROOT/logs/${name}.train.log"

  mkdir -p "$model_dir"

  echo
  echo "============================================================"
  echo "Training: $name"
  echo "  dataset: $dataset"
  echo "  output : $model_dir"
  echo "  window : $window_size"
  echo "  dim    : $emb_dim"
  echo "  epochs : $epochs"
  echo "  lr     : $lr"
  echo "  neg    : $neg"
  echo "============================================================"

  
  if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -f "WALL_TIME_SECONDS=%e\nMAX_RSS_KB=%M"       uv run python main.py train "$dataset"         --window-size "$window_size"         --emb-dim "$emb_dim"         --epochs "$epochs"         --learning-rate "$lr"         --neg-samples "$neg"         --seed "$SEED"         --output-dir "$model_dir"       2>&1 | tee "$log_file"
  else
    uv run python main.py train "$dataset"       --window-size "$window_size"       --emb-dim "$emb_dim"       --epochs "$epochs"       --learning-rate "$lr"       --neg-samples "$neg"       --seed "$SEED"       --output-dir "$model_dir"     2>&1 | tee "$log_file"
  fi
}

run_queries() {
  local name="$1"
  local model_dir="$OUT_ROOT/models/$name"
  local report_file="$OUT_ROOT/reports/${name}.qualitative.txt"

  echo
  echo "Running qualitative checks for: $name"

  {
    echo "MODEL: $name"
    echo "MODEL_DIR: $model_dir"
    echo

    echo "== most-similar(king) =="
    uv run python main.py use "$model_dir" most-similar king --topn 10 || true
    echo

    echo "== most-similar(city) =="
    uv run python main.py use "$model_dir" most-similar city --topn 10 || true
    echo

    echo "== similarity(king, queen) =="
    uv run python main.py use "$model_dir" similarity king queen || true
    echo

    echo "== similarity(paris, france) =="
    uv run python main.py use "$model_dir" similarity paris france || true
    echo

    echo "== analogy(man king woman) => man : king :: woman : ? =="
    uv run python main.py use "$model_dir" analogy man king woman --topn 5 || true
    echo

    echo "== analogy(paris france berlin) => paris : france :: berlin : ? =="
    uv run python main.py use "$model_dir" analogy paris france berlin --topn 5 || true
    echo
  } | tee "$report_file"
}


train_model "text8_100k_dim50_e3" "$OUT_ROOT/subsets/text8_100k.txt" 2 50 3 0.05 5
run_queries "text8_100k_dim50_e3"


train_model "text8_500k_dim50_e5" "$OUT_ROOT/subsets/text8_500k.txt" 2 50 5 0.05 5
run_queries "text8_500k_dim50_e5"


train_model "text8_1m_dim100_e5" "$OUT_ROOT/subsets/text8_1m.txt" 2 100 5 0.03 5
run_queries "text8_1m_dim100_e5"

echo
echo "Done. Results saved under: $OUT_ROOT"
echo "Models:   $OUT_ROOT/models/"
echo "Logs:     $OUT_ROOT/logs/"
echo "Reports:  $OUT_ROOT/reports/"
