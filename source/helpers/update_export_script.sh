#!/bin/bash

# Downloader for export_yolo26.py from the DeepStream-Yolo repository by Marcos Luciano.

URL="https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/utils/export_yolo26.py"
FILE="$(pwd)/../export.py"

curl -s -o "$FILE" "$URL"

if [ $? -ne 0 ]; then
    echo "Failed to download export.py from $URL."
    exit 1
fi

ATTRIBUTION=$(cat << 'EOF'
# ==============================================================================
# DISCLAIMER AND ATTRIBUTION
# Original Author: Marcos Luciano (marcoslucianops)
# Original Repository: https://github.com/marcoslucianops/DeepStream-Yolo
# License: MIT License
#
# This file, https://raw.githubusercontent.com/marcoslucianops/DeepStream-Yolo/master/utils/export_yolo26.py, 
# was automatically downloaded and may contain modifications.
# Please refer to the original repository for the official, up-to-date version.
# ==============================================================================
EOF
)

{
    echo "$ATTRIBUTION"
    echo ""
    cat "$FILE"
} > "${FILE}.tmp"

mv "${FILE}.tmp" "$FILE"

echo "export.py updated successfully from $URL with attributions added."