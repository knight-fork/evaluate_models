#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "usage: bash download.sh <google_drive_folder_link>"
    exit 1
fi
LINK="$1"

cd "$(dirname "$0")"

pip install -q gdown

gdown --folder "$LINK" -O ./checkpoints

ZIP="$(find ./checkpoints -name 'imagenet100_val5000.zip' | head -1)"
if [ -n "$ZIP" ]; then
    unzip -q -o "$ZIP" -d ./
fi
