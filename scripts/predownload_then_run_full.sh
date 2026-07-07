#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/hab_agentic_brain_physionet_20260706}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-4}"
RUNS="${RUNS:-04 08 12}"

cd "$PROJECT_DIR"

mkdir -p logs results data/mne/MNE-eegbci-data/files/eegmmidb/1.0.0

URL_LIST="logs/physionet_eegbci_urls.txt"
: > "$URL_LIST"
for subject in $(seq 1 109); do
  subject_id=$(printf "S%03d" "$subject")
  for run in $RUNS; do
    echo "https://physionet.org/files/eegmmidb/1.0.0/${subject_id}/${subject_id}R${run}.edf" >> "$URL_LIST"
  done
done

wget \
  --continue \
  --tries=5 \
  --timeout=30 \
  --waitretry=3 \
  --input-file="$URL_LIST" \
  --directory-prefix="data/mne/MNE-eegbci-data/files/eegmmidb/1.0.0" \
  --no-host-directories \
  --cut-dirs=3 \
  --force-directories \
  --recursive=false \
  --level=1 \
  --quiet \
  --show-progress \
  --progress=dot:giga \
  --execute robots=off \
  --random-wait \
  --limit-rate=0

python3 code/run_full_physionet_experiment.py \
  --config code/config_full_physionet.json \
  --result-dir results/full_physionet_loso
