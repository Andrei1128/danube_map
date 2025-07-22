#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import re

def extract_coordinates_from_filename(filename):
    """
    Extract coordinates from SVG filename in format:
    tile_{min_lat}_{min_lon}_{max_lat}_{max_lon}.svg
    """
    pattern = r'tile_([0-9.]+)_([0-9.]+)_([0-9.]+)_([0-9.]+)\.svg'
    match = re.match(pattern, filename)

    if match:
        min_lat = float(match.group(1))
        min_lon = float(match.group(2))
        max_lat = float(match.group(3))
        max_lon = float(match.group(4))

        return {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon
        }
    return None

def main():
    bathymetry_tiles_dir = "bathymetry_tiles"
    output_file = "tiles_metadata.json"

    if not os.path.exists(bathymetry_tiles_dir):
        print(f"Error: Directory '{bathymetry_tiles_dir}' not found")
        return

    files = glob.glob(os.path.join(bathymetry_tiles_dir, "*.svg"))

    if not files:
        print(f"No SVG files found in '{bathymetry_tiles_dir}'")
        return

    coordinates_data = []

    for path in files:
        filename = os.path.basename(path)
        coordinates = extract_coordinates_from_filename(filename)

        if coordinates:
            coordinates_data.append({
                "filename": filename,
                "coordinates": coordinates
            })
        else:
            print(f"Warning: Could not extract coordinates from filename: {filename}")

    with open(Path(bathymetry_tiles_dir) / output_file, 'w') as f:
        json.dump(coordinates_data, f, indent=2)

    print(f"Successfully extracted coordinates from {len(coordinates_data)} SVG files")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()