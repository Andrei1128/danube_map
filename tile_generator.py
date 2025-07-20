import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import gc
import json
import psutil
from math import floor, ceil
from scipy.interpolate import griddata

class BathymetryTiler:
    def __init__(self, csv_file, tile_size_deg=0.01, output_dir='tiles', overlap_factor=0.1):
        self.csv_file = csv_file
        self.tile_size = tile_size_deg
        self.overlap_factor = overlap_factor
        self.overlap_size = tile_size_deg * overlap_factor
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        print("Analyzing data bounds...")
        self._calculate_bounds()

        self.tile_min_x = floor(self.min_lon / self.tile_size)
        self.tile_max_x = ceil(self.max_lon / self.tile_size)
        self.tile_min_y = floor(self.min_lat / self.tile_size)
        self.tile_max_y = ceil(self.max_lat / self.tile_size)

        self.n_tiles_x = self.tile_max_x - self.tile_min_x
        self.n_tiles_y = self.tile_max_y - self.tile_min_y

        print(f"Tile grid: {self.n_tiles_x} x {self.n_tiles_y} = {self.n_tiles_x * self.n_tiles_y} tiles")

    def _calculate_bounds(self):
        data = pd.read_csv(self.csv_file)
        data = data[data['bathymetry'] >= 0]

        total_points = len(pd.read_csv(self.csv_file))
        water_points = len(data)

        self.min_lat = data['lat'].min()
        self.max_lat = data['lat'].max()
        self.min_lon = data['lon'].min()
        self.max_lon = data['lon'].max()

        print(f"Analyzed {water_points} water points from {total_points} total points")
        print(f"Bounds: Lat [{self.min_lat:.4f}, {self.max_lat:.4f}], Lon [{self.min_lon:.4f}, {self.max_lon:.4f}]")

    def get_tile_bounds(self, tile_x, tile_y, with_overlap=False):
        min_lon = tile_x * self.tile_size
        max_lon = (tile_x + 1) * self.tile_size
        min_lat = tile_y * self.tile_size
        max_lat = (tile_y + 1) * self.tile_size

        if with_overlap:
            min_lon -= self.overlap_size
            max_lon += self.overlap_size
            min_lat -= self.overlap_size
            max_lat += self.overlap_size

        return min_lon, max_lon, min_lat, max_lat

    def get_tile_data(self, tile_x, tile_y, with_overlap=True):
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y, with_overlap)

        data = pd.read_csv(self.csv_file)
        data = data[data['bathymetry'] >= 0]

        mask = (
            (data['lon'] >= min_lon) & (data['lon'] < max_lon) &
            (data['lat'] >= min_lat) & (data['lat'] < max_lat)
        )

        return data[mask]

    def create_tile_svg(self, tile_x, tile_y, resolution=800):
        # Calculate aspect ratio correction for longitude at this latitude
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y, with_overlap=False)
        center_lat = (min_lat + max_lat) / 2
        lon_correction = np.cos(np.radians(center_lat))

        # Adjust resolution for longitude to maintain proper aspect ratio
        resolution_lon = int(resolution * lon_correction)
        resolution_lat = resolution
        tile_data = self.get_tile_data(tile_x, tile_y, with_overlap=True)

        if len(tile_data) == 0:
            return

        min_lon_overlap, max_lon_overlap, min_lat_overlap, max_lat_overlap = self.get_tile_bounds(tile_x, tile_y, with_overlap=True)

        lon_grid_overlap = np.linspace(min_lon_overlap, max_lon_overlap, int(resolution_lon * (1 + 2 * self.overlap_factor)))
        lat_grid_overlap = np.linspace(min_lat_overlap, max_lat_overlap, int(resolution_lat * (1 + 2 * self.overlap_factor)))
        lon_mesh_overlap, lat_mesh_overlap = np.meshgrid(lon_grid_overlap, lat_grid_overlap)

        points = tile_data[['lon', 'lat']].values
        values = tile_data['bathymetry'].values

        # Need at least 4 points for triangulation
        if len(points) < 4:
            return

        z_grid_overlap = griddata(points, values, (lon_mesh_overlap, lat_mesh_overlap), method='linear')

        # Keep NaN values for areas without data (for transparency)
        # Only fill small gaps within the data area using nearest neighbor
        nan_mask = np.isnan(z_grid_overlap)
        if np.any(nan_mask) and np.any(~nan_mask):
            # Use a conservative approach: only fill NaNs that are surrounded by valid data
            from scipy.ndimage import binary_dilation
            valid_mask = ~nan_mask
            # Slightly expand the valid area to fill small interpolation gaps
            expanded_valid = binary_dilation(valid_mask, iterations=2)
            fill_mask = nan_mask & expanded_valid

            if np.any(fill_mask):
                z_grid_nearest = griddata(points, values, (lon_mesh_overlap, lat_mesh_overlap), method='nearest')
                z_grid_overlap[fill_mask] = z_grid_nearest[fill_mask]
                del z_grid_nearest

        # Apply gaussian filter for smoothing on the overlapped data
        from scipy.ndimage import gaussian_filter
        z_smoothed_overlap = gaussian_filter(z_grid_overlap, sigma=5.0)

        # Now crop to exact tile boundaries
        overlap_pixels_lon = int(resolution_lon * self.overlap_factor)
        overlap_pixels_lat = int(resolution_lat * self.overlap_factor)
        core_start_x = overlap_pixels_lon
        core_end_x = z_smoothed_overlap.shape[1] - overlap_pixels_lon
        core_start_y = overlap_pixels_lat
        core_end_y = z_smoothed_overlap.shape[0] - overlap_pixels_lat

        z_smoothed = z_smoothed_overlap[core_start_y:core_end_y, core_start_x:core_end_x]

        # Calculate figure size to maintain proper aspect ratio
        fig_width = 8
        fig_height = fig_width / lon_correction
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

        # Create coordinate grids for the exact tile area that match z_smoothed dimensions
        lon_grid = np.linspace(min_lon, max_lon, z_smoothed.shape[1])
        lat_grid = np.linspace(min_lat, max_lat, z_smoothed.shape[0])
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Create smooth filled contour plot with blue colormap
        contour_levels = np.arange(0, 39, 1)
        ax.contourf(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                   cmap='Blues', alpha=0.8, corner_mask=False)

        # Add smooth contour lines with enhanced smoothing
        cs_lines = ax.contour(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                             colors='black', linewidths=0.9, alpha=0.9,
                             corner_mask=False, antialiased=True)

        # Add labels to contour lines
        ax.clabel(cs_lines, inline=True, fontsize=12, fmt='%d', colors='black')

        # Set aspect ratio and limits
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect(1.0/lon_correction)  # Correct aspect ratio for latitude

        # Remove axes and margins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        plt.tight_layout(pad=0)

        svg_filename = os.path.join(self.output_dir, f'tile_{min_lat:.6f}_{min_lon:.6f}_{max_lat:.6f}_{max_lon:.6f}.svg')

        plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0,
                   transparent=True, facecolor='none')
        plt.close(fig)

        # Create metadata for this tile
        tile_metadata = {
            "filename": os.path.basename(svg_filename),
            "coordinates": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon
            }
        }

        # Explicit memory cleanup
        del (lon_mesh, lat_mesh, z_smoothed, z_smoothed_overlap, z_grid_overlap,
             lon_mesh_overlap, lat_mesh_overlap, points, values, tile_data,
             lon_grid_overlap, lat_grid_overlap, lon_grid, lat_grid)

        # Clean up any matplotlib objects
        plt.clf()
        plt.cla()
        gc.collect()

        return tile_metadata

    def _get_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB

    def generate_all_tiles(self, resolution=800, batch_size=50):
        total_tiles = (self.tile_max_x - self.tile_min_x) * (self.tile_max_y - self.tile_min_y)
        processed = 0
        all_metadata = []

        print(f"Starting tile generation. Initial memory usage: {self._get_memory_usage():.1f} MB")

        for tile_x in range(self.tile_min_x, self.tile_max_x):
            for tile_y in range(self.tile_min_y, self.tile_max_y):
                tile_metadata = self.create_tile_svg(tile_x, tile_y, resolution)
                if tile_metadata:
                    all_metadata.append(tile_metadata)
                processed += 1

                # Force garbage collection every batch_size tiles
                if processed % batch_size == 0:
                    gc.collect()
                    memory_mb = self._get_memory_usage()
                    print(f"Progress: {processed}/{total_tiles} tiles ({processed/total_tiles*100:.1f}%) - Memory: {memory_mb:.1f} MB")

        # Save all metadata to JSON file
        metadata_file = os.path.join(self.output_dir, 'tiles_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        final_memory = self._get_memory_usage()
        print(f"Completed all {processed} tiles - Final memory usage: {final_memory:.1f} MB")
        print(f"Metadata saved to {metadata_file}")
        return all_metadata

if __name__ == "__main__":
    tiler = BathymetryTiler(
        csv_file='bathymetry_Danube.csv',
        tile_size_deg=0.01,
        output_dir='bathymetry_tiles'
    )

    metadata = tiler.generate_all_tiles(resolution=1000, batch_size=5)