#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_dir="${root_dir}/data"
target_dir="${download_dir}/m5-forecasting-accuracy"
archive="${download_dir}/m5-forecasting-accuracy.zip"

command -v kaggle >/dev/null 2>&1 || {
  echo "Kaggle CLI is required. Install it with: pip install kaggle" >&2
  exit 1
}

mkdir -p "${target_dir}"
kaggle competitions download -c m5-forecasting-accuracy -p "${download_dir}"
unzip -o "${archive}" -d "${target_dir}"
echo "M5 data extracted to ${target_dir}"

