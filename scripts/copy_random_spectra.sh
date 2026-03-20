#!/bin/bash

# --- CONFIGURATION ---
SOURCE_DIR="/Users/pxm588@student.bham.ac.uk/Desktop/snid/cfa_pipeline/all_spectra_dereddened_snidded_supersnid"
DEST_DIR="data/spectra/random"
NUM_FILES=100
# ---------------------

# 1. Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# 2. Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Selecting $NUM_FILES random files from $SOURCE_DIR..."

# 3. Find files, shuffle using python (native on macOS), take the first 100, and copy
find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.flm" | \
python3 -c "import sys, random; lines = sys.stdin.readlines(); random.shuffle(lines); sys.stdout.write(''.join(lines[:$NUM_FILES]))" | \
while read -r file; do
    cp "$file" "$DEST_DIR/"
done

echo "Successfully copied $(ls -1 "$DEST_DIR" | wc -l) files to $DEST_DIR."
