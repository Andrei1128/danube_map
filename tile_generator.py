import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from math import floor, ceil
from scipy.interpolate import griddata

class BathymetryTiler:
    def __init__(self, csv_file, tile_size_deg=0.01, output_dir='tiles'):
        self.tile_size = tile_size_deg
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        print("Loading data...")
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df)} points")

        self.df = self.df[self.df['bathymetry'] >= 0]
        print(f"Filtered to {len(self.df)} water points (removed land with negative depths)")

        self.min_lat = self.df['lat'].min()
        self.max_lat = self.df['lat'].max()
        self.min_lon = self.df['lon'].min()
        self.max_lon = self.df['lon'].max()

        print(f"Bounds: Lat [{self.min_lat:.4f}, {self.max_lat:.4f}], Lon [{self.min_lon:.4f}, {self.max_lon:.4f}]")

        self.tile_min_x = floor(self.min_lon / self.tile_size)
        self.tile_max_x = ceil(self.max_lon / self.tile_size)
        self.tile_min_y = floor(self.min_lat / self.tile_size)
        self.tile_max_y = ceil(self.max_lat / self.tile_size)

        self.n_tiles_x = self.tile_max_x - self.tile_min_x
        self.n_tiles_y = self.tile_max_y - self.tile_min_y

        print(f"Tile grid: {self.n_tiles_x} x {self.n_tiles_y} = {self.n_tiles_x * self.n_tiles_y} tiles")

    def get_tile_bounds(self, tile_x, tile_y):
        min_lon = tile_x * self.tile_size
        max_lon = (tile_x + 1) * self.tile_size
        min_lat = tile_y * self.tile_size
        max_lat = (tile_y + 1) * self.tile_size
        return min_lon, max_lon, min_lat, max_lat

    def get_tile_data(self, tile_x, tile_y):
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y)

        mask = (
            (self.df['lon'] >= min_lon) & (self.df['lon'] < max_lon) &
            (self.df['lat'] >= min_lat) & (self.df['lat'] < max_lat)
        )

        return self.df[mask]

    def create_tile_svg(self, tile_x, tile_y, resolution=800):
        tile_data = self.get_tile_data(tile_x, tile_y)

        if len(tile_data) == 0:
            return

        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y)

        lon_grid = np.linspace(min_lon, max_lon, resolution)
        lat_grid = np.linspace(min_lat, max_lat, resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        points = tile_data[['lon', 'lat']].values
        values = tile_data['bathymetry'].values

        # Need at least 4 points for triangulation
        if len(points) < 4:
            return

        z_grid = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

        # Keep NaN values for areas without data (for transparency)
        # Only fill small gaps within the data area using nearest neighbor
        nan_mask = np.isnan(z_grid)
        if np.any(nan_mask) and np.any(~nan_mask):
            # Use a conservative approach: only fill NaNs that are surrounded by valid data
            from scipy.ndimage import binary_dilation
            valid_mask = ~nan_mask
            # Slightly expand the valid area to fill small interpolation gaps
            expanded_valid = binary_dilation(valid_mask, iterations=2)
            fill_mask = nan_mask & expanded_valid

            if np.any(fill_mask):
                z_grid_nearest = griddata(points, values, (lon_mesh, lat_mesh), method='nearest')
                z_grid[fill_mask] = z_grid_nearest[fill_mask]

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        #Apply gaussian filter for smoothing
        from scipy.ndimage import gaussian_filter
        z_smoothed = gaussian_filter(z_grid, sigma=5.0)

        # Create smooth filled contour plot with blue colormap
        contour_levels = np.arange(0, 38, 1)
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
        ax.set_aspect('equal')

        # Remove axes and margins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        plt.tight_layout(pad=0)

        svg_filename = os.path.join(self.output_dir, f'tile_{tile_x}_{tile_y}.svg')

        plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0,
                   transparent=True, facecolor='none')
        plt.close()

        print(f"Processed tile ({tile_x}, {tile_y})...")

    def generate_all_tiles(self, resolution=800):
        for tile_x in range(self.tile_min_x, self.tile_max_x):
            for tile_y in range(self.tile_min_y, self.tile_max_y):
                self.create_tile_svg(tile_x, tile_y, resolution)

if __name__ == "__main__":
    tiler = BathymetryTiler(
        csv_file='bathymetry_Danube.csv',
        tile_size_deg=0.01,
        output_dir='bathymetry_tiles'
    )

    metadata = tiler.generate_all_tiles(resolution=2500)