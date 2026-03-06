#!/bin/bash

# --- CONFIGURATION ---
TEMPLATE_DIR="/home/sniduser/snid-5.0/templates"
TEMPLATE_EXT="lnw"
PARAM_FILE="cfasnIa_param.dat"
SUBTYPE_FILE="cfa_subtypes_and_z_summary.csv"
# ---------------------

if [ ! -f "$PARAM_FILE" ] || [ ! -f "$SUBTYPE_FILE" ]; then
    echo "Error: Required data files missing."
    exit 1
fi

for file in *.flm ; do
    [ -e "$file" ] || continue
    echo "--- Processing file: $file ---"

    # --- 1. BRANCHING LOGIC FOR NAMING CONVENTIONS ---
    
    # Example 1 (Standard): sn2007fc-20070711.47-fast.flm
    # Example 2 (SNF):      snf20080720-001-20080801.44-fast.flm

    if [[ "$file" == snf* ]] || [[ "$file" == sne* ]]; then
        # SNF/SNE LOGIC
        # 1. Base name for 'avoid' and 'template': snf20080720-001
        avoid_id=$(echo "$file" | cut -d'-' -f1,2)
        
        # 2. Lookup for param file (Uppercase): SNF20080720-001
        param_lookup=$(echo "$avoid_id" | tr '[:lower:]' '[:upper:]')
        
        # 3. Key for Subtype CSV: snf20080720-001-20080801
        # We take everything before the first "."
        csv_key=${file%%.*}
    else
        # STANDARD SN LOGIC
        # 1. Base name for 'avoid' and 'template': sn2007fc
        avoid_id=${file%%-*}
        
        # 2. Lookup for param file: 2007fc
        param_lookup=${avoid_id#sn}
        
        # 3. Key for Subtype CSV: sn2007fc-20070711
        csv_key=${file%%.*}
    fi

    # --- 2. LOOKUP REDSHIFT ---
    # Matches "SNF20080720-001 " or "2007fc " at start of line
    redshift=$(grep -m 1 "^${param_lookup}[[:space:]]" "$PARAM_FILE" | awk '{print $2}')
    z_arg=""
    if [ -n "$redshift" ]; then
        z_arg="forcez=$redshift"
        echo "Found Redshift for $param_lookup: $redshift"
    else
        echo "WARNING: Redshift not found for '$param_lookup' in $PARAM_FILE"
    fi

    # --- 3. LOOKUP SUBTYPE ---
    # Matches "snf20080720-001-20080801," or "sn2007fc-20070711,"
    subtype=$(grep -m 1 "^${csv_key}," "$SUBTYPE_FILE" | cut -d',' -f2 | tr -d '\r')
    subtype_arg=""
    if [ -n "$subtype" ]; then
        subtype_arg="usetype=$subtype"
        echo "Found Subtype for $csv_key: $subtype"
    else
        echo "WARNING: Subtype not found for '$csv_key' in $SUBTYPE_FILE"
    fi

    # --- 4. TEMPLATE AVOID CHECK ---
    template_file="$TEMPLATE_DIR/${avoid_id}.${TEMPLATE_EXT}"
    avoid_arg=""
    if [ -f "$template_file" ]; then
        avoid_arg="avoid=$avoid_id"
        echo "Template for '$avoid_id' exists. Adding avoid argument."
    fi

    # --- 5. RUN SNID ---
    # Running with fout=10 to save the cross-correlation results
    echo "Executing: snid $avoid_arg $z_arg $subtype_arg inter=0 plot=0 verbose=0\"$file\""
    snid $avoid_arg $z_arg $subtype_arg inter=0 plot=0 verbose=0 fout=10 "$file"
    
    echo 
done

echo "--- Script finished. ---"