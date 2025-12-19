#!/bin/bash
# Smart pull - only pulls if image doesn't exist locally

set -e

IMAGE_LIST_FILE=$1
FORCE_PULL=${2:-false}  # Optional: set to "true" to force re-pull

if [ -z "$IMAGE_LIST_FILE" ]; then
    echo "Usage: $0 <image_list_file> [force_pull]"
    echo "Example: $0 tdd_bench_images.txt"
    echo "         $0 tdd_bench_images.txt true  # Force re-pull existing images"
    exit 1
fi

if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Error: Image list file '$IMAGE_LIST_FILE' not found"
    exit 1
fi

echo "Pulling Docker images from $IMAGE_LIST_FILE..."
echo "Force pull: $FORCE_PULL"
echo "================================================"

TOTAL_IMAGES=$(wc -l < "$IMAGE_LIST_FILE")
CURRENT=0
FAILED=0
SUCCESS=0
SKIPPED=0

# Read each line from the file and pull the docker image
while IFS= read -r image; do
    # Skip empty lines
    if [ -z "$image" ]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL_IMAGES] Processing: $image"

    # Check if image exists locally
    if docker image inspect "$image" >/dev/null 2>&1; then
        if [ "$FORCE_PULL" = "true" ]; then
            echo "  Image exists locally, but force pull is enabled"
        else
            echo "  ✓ Image already exists locally - SKIPPING"
            SKIPPED=$((SKIPPED + 1))

            # Still tag the renamed version if it doesn't exist
            renamed_image=$(echo "$image" | sed 's|.*/||; s/_s_/__/g')
            if ! docker image inspect "$renamed_image" >/dev/null 2>&1; then
                docker tag "$image" "$renamed_image"
                echo "    Tagged as: $renamed_image"
            fi
            continue
        fi
    fi

    # Pull the image
    echo "  Pulling from registry..."
    if docker pull "$image"; then
        echo "  ✓ Successfully pulled: $image"
        SUCCESS=$((SUCCESS + 1))

        # Tag with renamed version
        renamed_image=$(echo "$image" | sed 's|.*/||; s/_s_/__/g')
        docker tag "$image" "$renamed_image"
        echo "    Tagged as: $renamed_image"
    else
        echo "  ✗ Failed to pull: $image"
        FAILED=$((FAILED + 1))
    fi
done < "$IMAGE_LIST_FILE"

echo ""
echo "================================================"
echo "Pull Summary:"
echo "  Total: $TOTAL_IMAGES"
echo "  Success: $SUCCESS"
echo "  Skipped (already exists): $SKIPPED"
echo "  Failed: $FAILED"
echo "================================================"

if [ $FAILED -gt 0 ]; then
    echo "WARNING: Some images failed to pull."
    exit 1
fi

echo "Done!"
