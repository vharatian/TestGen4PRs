#!/bin/bash
# Complete workflow: Generate image list from JSON and pull all images

set -e

JSON_FILE=$1
IMAGE_PREFIX=${2:-"docker.io/xingyaoww/"}

if [ -z "$JSON_FILE" ]; then
    echo "Usage: $0 <json_dataset_file> [image_prefix]"
    echo ""
    echo "Example:"
    echo "  $0 data/tdd_bench_first10.json"
    echo "  $0 data/tdd_bench_first10.json ghcr.io/openhands/"
    echo ""
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file '$JSON_FILE' not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_NAME=$(basename "$JSON_FILE" .json)
IMAGE_LIST_FILE="${SCRIPT_DIR}/${DATASET_NAME}_images.txt"

echo "================================================"
echo "Setting up Docker images for: $JSON_FILE"
echo "Image prefix: $IMAGE_PREFIX"
echo "================================================"
echo ""

# Step 1: Generate image list
echo "[Step 1/2] Generating image list..."
python3 "${SCRIPT_DIR}/json_to_image_list.py" "$JSON_FILE" "$IMAGE_LIST_FILE" "$IMAGE_PREFIX"

if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Error: Failed to generate image list"
    exit 1
fi

echo ""
echo "Image list generated at: $IMAGE_LIST_FILE"
echo ""

# Step 2: Pull images (using smart pull to skip existing images)
echo "[Step 2/2] Pulling Docker images..."
bash "${SCRIPT_DIR}/pull_images_smart.sh" "$IMAGE_LIST_FILE"

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
