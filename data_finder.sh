#!/bin/bash

# Usage: ./data_finder.sh <nside> <phi_scale> [--lensed|--unlensed] [--old]
# Examples:
#   ./data_finder.sh 64 1.0 --lensed         # Search data/logs only
#   ./data_finder.sh 64 1.0 --lensed --old   # Search both data/logs and data/old/logs
#   ./data_finder.sh 64 all --unlensed       # Search all phi_scales in data/logs
#   ./data_finder.sh 64 all --unlensed --old # Search all phi_scales in both directories

if [ $# -lt 2 ]; then
    echo "Usage: $0 <nside> <phi_scale> [--lensed|--unlensed] [--old]"
    echo ""
    echo "Arguments:"
    echo "  nside          : Resolution parameter (e.g., 64)"
    echo "  phi_scale      : Scale parameter (e.g., 1.0) or 'all' for all phi_scales"
    echo "  lensing_option : --lensed (default) or --unlensed"
    echo "  --old          : Include data/old/logs/trainer/ directory (default: only data/logs/trainer/)"
    exit 1
fi

nside=$1
phi_scale=$2
shift 2

# Parse remaining arguments for lensing option and --old flag
lensing_option="--lensed"
search_old=false

for arg in "$@"; do
    if [ "$arg" == "--lensed" ]; then
        lensing_option="--lensed"
    elif [ "$arg" == "--unlensed" ]; then
        lensing_option="--unlensed"
    elif [ "$arg" == "--old" ]; then
        search_old=true
    else
        echo "Error: Unknown argument '$arg'"
        exit 1
    fi
done

# Determine trainer module based on lensing option
if [ "$lensing_option" == "--unlensed" ]; then
    trainer_module="mlpng.trainer_unlensed"
elif [ "$lensing_option" == "--lensed" ]; then
    trainer_module="mlpng.trainer_lensed"
else
    echo "Error: Invalid lensing option: $lensing_option"
    exit 1
fi

# Build search directories
search_dirs="data/logs/trainer/*"
if [ "$search_old" == true ]; then
    search_dirs="data/logs/trainer/* data/old/logs/trainer/*"
fi

# Function to search and print results for a specific phi_scale value
search_phi_scale() {
    local search_phi_scale=$1
    local found_any=0

    # Search in specified directories
    for log_file in $search_dirs; do
        if [ -f "$log_file" ]; then
            # Check if file contains the specific nside, phi_scale, and trainer module
            if grep -q "\"nside\": $nside" "$log_file" && \
               grep -q "\"phi_scale\": $search_phi_scale" "$log_file" && \
               grep -q "Running trainer with module $trainer_module" "$log_file"; then

                found_any=1
                echo "=== Found in: $log_file ==="

                # Extract plot folder from the log file
                plot_folder=$(grep -oP 'Saved training history plot: \K[^ ]+' "$log_file" | sed 's|/[^/]*$||' | head -1)

                # Display metadata
                echo "  nside: $nside"
                echo "  phi_scale: $search_phi_scale"
                echo "  module: $trainer_module"
                if [ ! -z "$plot_folder" ]; then
                    echo "  plot folder: $plot_folder"
                fi
                echo ""

                # Extract from "Test Set Performance Summary" to just before the first timestamp log line
                sed -n '/Test Set Performance Summary/,/^[0-9]\{2\}-[A-Za-z]/p' "$log_file" | head -n -1
                echo ""
            fi
        fi
    done

    return $found_any
}

# Handle "all" case for phi_scale
if [ "$phi_scale" == "all" ]; then
    found_any=0

    # Alternative: Get phi_scales from files that match both nside and trainer module
    temp_dir=$(mktemp -d)
    for log_file in $search_dirs; do
        if [ -f "$log_file" ]; then
            if grep -q "\"nside\": $nside" "$log_file" && \
               grep -q "Running trainer with module $trainer_module" "$log_file"; then
                grep "\"phi_scale\":" "$log_file" >> "$temp_dir/phi_scales.txt" 2>/dev/null
            fi
        fi
    done

    if [ -f "$temp_dir/phi_scales.txt" ]; then
        unique_phi_scales=$(sed 's/.*"phi_scale": //; s/[,}].*//' "$temp_dir/phi_scales.txt" | sort -u)
    fi

    rm -rf "$temp_dir"

    # Search for each unique phi_scale
    for ps in $unique_phi_scales; do
        if search_phi_scale "$ps"; then
            found_any=1
        fi
    done

    if [ $found_any -eq 0 ]; then
        echo "No log files found matching: nside=$nside, module=$trainer_module"
        exit 1
    fi
else
    # Search for specific phi_scale value
    if ! search_phi_scale "$phi_scale"; then
        echo "No log files found matching: nside=$nside, phi_scale=$phi_scale, module=$trainer_module"
        exit 1
    fi
fi
