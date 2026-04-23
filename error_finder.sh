#!/bin/bash

# Usage: ./error_finder.sh <nside> <phi_scale> [--reduced] [--old]
# Extracts RMSE values from mlpng.trainer_lensed logs for given nside+phiscale
# Searches for "Comparison: Full Range vs Restricted Range" table section
# Calculates mean, best (min), worst (max), and standard deviation
# Examples:
#   ./error_finder.sh 64 1.0                    # Full range (column 3: Full RMSE)
#   ./error_finder.sh 64 1.0 --reduced          # Reduced range (column 6: [-100,100] RMSE)
#   ./error_finder.sh 64 1.0 --reduced --old    # Reduced range + data/old/logs

if [ $# -lt 2 ]; then
    echo "Usage: $0 <nside> <phi_scale> [--reduced] [--old]"
    echo ""
    echo "Arguments:"
    echo "  nside     : Resolution parameter (e.g., 64)"
    echo "  phi_scale : Scale parameter (e.g., 1.0)"
    echo "  --reduced : Extract reduced range RMSE ([-100,100]) instead of full range (default: full range)"
    echo "  --old     : Include data/old/logs/trainer/ directory (default: only data/logs/trainer/)"
    exit 1
fi

nside=$1
phi_scale=$2
search_old=false
use_reduced=false

# Parse optional flags
for arg in "$3" "$4"; do
    if [ "$arg" == "--old" ]; then
        search_old=true
    elif [ "$arg" == "--reduced" ]; then
        use_reduced=true
    fi
done

# Build search directories
search_dirs="data/logs/trainer/*"
if [ "$search_old" == true ]; then
    search_dirs="data/logs/trainer/* data/old/logs/trainer/*"
fi

# Determine which RMSE column to extract
if [ "$use_reduced" == true ]; then
    rmse_col=6  # [-100,100] RMSE column
    rmse_label="Reduced Range [-100,100]"
else
    rmse_col=3  # Full RMSE column
    rmse_label="Full Range"
fi

# Temporary file for RMSE values
temp_file=$(mktemp)
trap "rm -f $temp_file" EXIT

# Search in specified directories
for log_file in $search_dirs; do
    if [ -f "$log_file" ]; then
        # Check if file contains the specific nside, phi_scale, and trainer module
        if grep -q "\"nside\": $nside" "$log_file" && \
           grep -q "\"phi_scale\": $phi_scale" "$log_file" && \
           grep -q "Running trainer with module mlpng.trainer_lensed" "$log_file"; then

            # Extract RMSE values from comparison table
            # Table appears after "Comparison: Full Range vs Restricted Range" header
            # Format:
            # Comparison: Full Range vs Restricted Range (per-shape filtering)
            # ===...===
            # Shape  Full N  Full RMSE  Full RMSE/σ  [-100,100] N  [-100,100] RMSE  [-100,100] RMSE/σ         σ
            # local    2000  58.656944     1.232679           198        48.615425           1.021656 47.584945
            # ===...===

            # Get lines after the comparison header, then extract data lines (skip header/separator)
            awk -v col=$rmse_col '
                /Comparison: Full Range vs Restricted Range/ {
                    found=1
                    count=0
                    next
                }
                found && count < 5 {
                    count++
                    # Skip "=" separator lines
                    if ($0 ~ /^=/) {
                        if (count >= 4) {
                            # Hit the closing "=" line, stop
                            found=0
                        }
                        next
                    }
                    # Skip empty lines and column header line
                    if (NF == 0 || $1 == "Shape") {
                        next
                    }
                    # Only extract RMSE for "local" shape (skip "all" or other shapes)
                    if ($1 != "local") {
                        next
                    }
                    # Extract specified column (Full RMSE or Reduced RMSE)
                    if (NF >= col) {
                        val = $col
                        if (val ~ /^[0-9]+(\.[0-9]+)?$/ && val > 0 && val < 10000) {
                            print val
                        }
                    }
                }
            ' "$log_file" >> "$temp_file"
        fi
    fi
done

# Check if we found any RMSE values
if [ ! -s "$temp_file" ]; then
    echo "No RMSE values found for nside=$nside, phi_scale=$phi_scale (range: $rmse_label)"
    exit 1
fi

# Read RMSE values into array
mapfile -t rmse_values < "$temp_file"
rmse_count=${#rmse_values[@]}

# Calculate statistics using awk (for floating point precision)
stats=$(awk '{
    sum += $1
    if (NR == 1) { min = $1; max = $1 }
    if ($1 < min) min = $1
    if ($1 > max) max = $1
    values[NR] = $1
}
END {
    if (NR > 0) {
        mean = sum / NR
        # Calculate standard deviation (sample std dev with N-1)
        sum_sq_diff = 0
        for (i=1; i<=NR; i++) {
            diff = values[i] - mean
            sum_sq_diff += diff * diff
        }
        if (NR > 1) {
            stddev = sqrt(sum_sq_diff / (NR - 1))
        } else {
            stddev = 0
        }
        printf "%.6f:%.6f:%.6f:%.6f\n", mean, min, max, stddev
    }
}' "$temp_file")

# Parse statistics
mean=$(echo "$stats" | cut -d: -f1)
best=$(echo "$stats" | cut -d: -f2)
worst=$(echo "$stats" | cut -d: -f3)
stddev=$(echo "$stats" | cut -d: -f4)

# Print all found RMSE values
echo "=== RMSE Values Found (nside=$nside, phi_scale=$phi_scale, $rmse_label) ==="
for val in "${rmse_values[@]}"; do
    echo "$val"
done
echo ""

# Print summary
echo "=== Summary ($rmse_label) ==="
echo "Mean RMSE: $mean"
echo "Best RMSE: $best"
echo "Worst RMSE: $worst"
echo "Std Dev: $stddev"
echo "Count: $rmse_count"
echo ""

# Print structured output for shell sourcing (compatible with bash eval)
echo "export MEAN_RMSE=\"$mean\""
echo "export BEST_RMSE=\"$best\""
echo "export WORST_RMSE=\"$worst\""
echo "export STD_DEV=\"$stddev\""
echo "export RMSE_COUNT=\"$rmse_count\""
