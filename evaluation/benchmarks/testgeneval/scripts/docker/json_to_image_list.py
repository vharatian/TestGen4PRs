#!/usr/bin/env python3
"""Generate a list of Docker images needed for a custom JSON dataset."""

import json
import sys


def json_to_image_list(json_file, output_file, image_prefix='docker.io/xingyaoww/'):
    """
    Generate a text file with Docker image names from a JSON dataset.

    Args:
        json_file: Path to the JSON dataset file
        output_file: Path to output text file with image names
        image_prefix: Docker registry prefix (default: docker.io/kdjain/)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        instances = data.get('instances', [data])
    else:
        instances = data

    images = set()  # Use set to avoid duplicates

    for instance in instances:
        # Get instance_id (try different field names)
        instance_id = instance.get('instance_id') or instance.get('id')

        if not instance_id:
            print(f"Warning: No instance_id found for entry: {instance.get('repo', 'unknown')}")
            continue

        # Convert __ to _s_ for Docker naming convention
        image_name = instance_id.replace('__', '_s_')

        # Build full image name
        full_image = f"{image_prefix.rstrip('/')}/sweb.eval.x86_64.{image_name}:latest"
        images.add(full_image)

    # Write to output file
    with open(output_file, 'w') as f:
        for image in sorted(images):
            f.write(f"{image}\n")

    print(f"Generated {len(images)} image names in {output_file}")
    return len(images)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python json_to_image_list.py <json_file> [output_file] [image_prefix]")
        print("Example: python json_to_image_list.py tdd_bench_first10.json tdd_bench_images.txt")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'custom_images.txt'
    image_prefix = sys.argv[3] if len(sys.argv) > 3 else 'docker.io/xingyaoww/'

    json_to_image_list(json_file, output_file, image_prefix)
