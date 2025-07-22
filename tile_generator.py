import gc
import json
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, gaussian_filter

@dataclass
class TilerConfig:
    """Configuration for the bathymetry tiler."""
    tile_size_deg: float = 0.01
    overlap_factor: float = 0.1
    resolution: int = 1000
    batch_size: int = 10
    smoothing_sigma: float = 5
    contour_max_depth: float = 40.0
    contour_step: float = 1.0
    fig_width: float = 8.0
    fig_dpi: int = 100,
    tile_opacity: float = 0.9,
    contour_opacity: float = 0.9

    @property
    def overlap_size(self) -> float:
        return self.tile_size_deg * self.overlap_factor


@dataclass
class DataBounds:
    """Data bounds container."""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    total_points: int
    water_points: int


class DataLoader:
    """Handles loading and filtering bathymetry data."""

    def __init__(self, csv_file: str | Path):
        self.csv_file = Path(csv_file)
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

    def calculate_bounds(self) -> DataBounds:
        """Calculate data bounds and statistics."""
        data = pd.read_csv(self.csv_file)

        total_points = len(pd.read_csv(self.csv_file))
        water_points = len(data)

        bounds = DataBounds(
            min_lat=data['lat'].min(),
            max_lat=data['lat'].max(),
            min_lon=data['lon'].min(),
            max_lon=data['lon'].max(),
            total_points=total_points,
            water_points=water_points
        )

        print(f"Analyzed {water_points} water points from {total_points} total points")
        print(f"Bounds: Lat [{bounds.min_lat:.4f}, {bounds.max_lat:.4f}], Lon [{bounds.min_lon:.4f}, {bounds.max_lon:.4f}]")

        return bounds

    def load_tile_data(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> pd.DataFrame:
        """Load data for a specific tile area."""
        data = pd.read_csv(self.csv_file)

        mask = (
            (data['lon'] >= min_lon) & (data['lon'] < max_lon) &
            (data['lat'] >= min_lat) & (data['lat'] < max_lat)
        )

        return data[mask]


class BathymetryTiler:
    """Generates SVG tiles from bathymetry data."""

    def __init__(self, csv_file: str | Path, output_dir: str | Path = 'tiles', config: Optional[TilerConfig] = None,
                 min_lat: Optional[float] = None, max_lat: Optional[float] = None,
                 min_lon: Optional[float] = None, max_lon: Optional[float] = None):
        self.config = config or TilerConfig()
        self.output_dir = Path(output_dir)
        self.data_loader = DataLoader(csv_file)

        self.output_dir.mkdir(exist_ok=True)

        print("Analyzing data bounds...")
        self.bounds = self.data_loader.calculate_bounds()

        # Use custom bounds if provided, otherwise use data bounds
        if min_lat is not None:
            print(f"  min_lat: {min_lat:.4f}")
            self.bounds.min_lat = min_lat

        if max_lat is not None:
            print(f"  max_lat: {max_lat:.4f}")
            self.bounds.max_lat = max_lat

        if min_lon is not None:
            print(f"  min_lon: {min_lon:.4f}")
            self.bounds.min_lon = min_lon

        if max_lon is not None:
            print(f"  max_lon: {max_lon:.4f}")
            self.bounds.max_lon = max_lon

        self.tile_min_x = floor(self.bounds.min_lon / self.config.tile_size_deg)
        self.tile_max_x = ceil(self.bounds.max_lon / self.config.tile_size_deg)
        self.tile_min_y = floor(self.bounds.min_lat / self.config.tile_size_deg)
        self.tile_max_y = ceil(self.bounds.max_lat / self.config.tile_size_deg)

        self.n_tiles_x = self.tile_max_x - self.tile_min_x
        self.n_tiles_y = self.tile_max_y - self.tile_min_y

        print(f"Tile grid: {self.n_tiles_x} x {self.n_tiles_y} = {self.n_tiles_x * self.n_tiles_y} tiles")


    def get_tile_bounds(self, tile_x: int, tile_y: int, with_overlap: bool = False) -> Tuple[float, float, float, float]:
        """Get the bounds of a tile."""
        min_lon = tile_x * self.config.tile_size_deg
        max_lon = (tile_x + 1) * self.config.tile_size_deg
        min_lat = tile_y * self.config.tile_size_deg
        max_lat = (tile_y + 1) * self.config.tile_size_deg

        if with_overlap:
            overlap = self.config.overlap_size
            min_lon -= overlap
            max_lon += overlap
            min_lat -= overlap
            max_lat += overlap

        return min_lon, max_lon, min_lat, max_lat

    def get_tile_data(self, tile_x: int, tile_y: int, with_overlap: bool = True) -> pd.DataFrame:
        """Get data for a specific tile."""
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y, with_overlap)
        return self.data_loader.load_tile_data(min_lon, max_lon, min_lat, max_lat)

    def _calculate_resolution(self, center_lat: float, base_resolution: int) -> Tuple[int, int]:
        """Calculate longitude and latitude resolution with aspect ratio correction."""
        lon_correction = np.cos(np.radians(center_lat))
        resolution_lon = int(base_resolution * lon_correction)
        resolution_lat = base_resolution
        return resolution_lon, resolution_lat

    def _interpolate_bathymetry(self, tile_data: pd.DataFrame, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> np.ndarray:
        """Interpolate bathymetry data onto a grid."""
        points = tile_data[['lon', 'lat']].values
        values = tile_data['bathymetry'].values

        if len(points) < 4:
            raise ValueError("Need at least 4 points for triangulation")

        z_grid = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

        # Fill small gaps using conservative approach
        nan_mask = np.isnan(z_grid)
        if np.any(nan_mask) and np.any(~nan_mask):
            valid_mask = ~nan_mask
            expanded_valid = binary_dilation(valid_mask, iterations=2)
            fill_mask = nan_mask & expanded_valid

            if np.any(fill_mask):
                z_grid_nearest = griddata(points, values, (lon_mesh, lat_mesh), method='nearest')
                z_grid[fill_mask] = z_grid_nearest[fill_mask]

        return z_grid

    def _smooth_and_crop(self, z_grid: np.ndarray, resolution_lon: int, resolution_lat: int) -> np.ndarray:
        """Apply smoothing and crop to exact tile boundaries."""
        z_smoothed = gaussian_filter(z_grid, sigma=self.config.smoothing_sigma)

        # Crop to exact tile boundaries
        overlap_pixels_lon = int(resolution_lon * self.config.overlap_factor) - 2
        overlap_pixels_lat = int(resolution_lat * self.config.overlap_factor) - 2

        core_start_x = overlap_pixels_lon
        core_end_x = z_smoothed.shape[1] - overlap_pixels_lon
        core_start_y = overlap_pixels_lat
        core_end_y = z_smoothed.shape[0] - overlap_pixels_lat

        return z_smoothed[core_start_y:core_end_y, core_start_x:core_end_x]

    def _create_plot(self, z_smoothed: np.ndarray, min_lon: float, max_lon: float, min_lat: float, max_lat: float, lon_correction: float) -> Tuple[plt.Figure, plt.Axes]:
        """Create the matplotlib plot for the tile."""
        fig_height = self.config.fig_width / lon_correction
        fig, ax = plt.subplots(figsize=(self.config.fig_width, fig_height), dpi=self.config.fig_dpi)

        # Create coordinate grids
        lon_grid = np.linspace(min_lon, max_lon, z_smoothed.shape[1])
        lat_grid = np.linspace(min_lat, max_lat, z_smoothed.shape[0])
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        # Create contour plots
        contour_levels = np.arange(0, self.config.contour_max_depth, self.config.contour_step)
        ax.contourf(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                   cmap='Blues', alpha=self.config.tile_opacity, corner_mask=False)

        cs_lines = ax.contour(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                             colors='black', linewidths=0.9, alpha=self.config.contour_opacity,
                             corner_mask=False, antialiased=True)

        ax.clabel(cs_lines, inline=True, fontsize=12, fmt='%d', colors='black')

        # Configure plot appearance
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect(1.0/lon_correction)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        plt.tight_layout(pad=0)

        return fig, ax

    def _cleanup_memory(self, *objects) -> None:
        """Clean up memory by deleting objects and running garbage collection."""
        for obj in objects:
            del obj
        plt.clf()
        plt.cla()
        gc.collect()

    def create_tile_svg(self, tile_x: int, tile_y: int, resolution: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Create an SVG tile for the given coordinates."""
        if resolution is None:
            resolution = self.config.resolution

        try:
            min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y, with_overlap=False)
            center_lat = (min_lat + max_lat) / 2
            lon_correction = np.cos(np.radians(center_lat))

            resolution_lon, resolution_lat = self._calculate_resolution(center_lat, resolution)
            tile_data = self.get_tile_data(tile_x, tile_y, with_overlap=True)

            if len(tile_data) == 0:
                return None

            min_lon_overlap, max_lon_overlap, min_lat_overlap, max_lat_overlap = self.get_tile_bounds(tile_x, tile_y, with_overlap=True)

            # Create overlapped grid
            lon_grid_overlap = np.linspace(min_lon_overlap, max_lon_overlap,
                                         int(resolution_lon * (1 + 2 * self.config.overlap_factor)))
            lat_grid_overlap = np.linspace(min_lat_overlap, max_lat_overlap,
                                         int(resolution_lat * (1 + 2 * self.config.overlap_factor)))
            lon_mesh_overlap, lat_mesh_overlap = np.meshgrid(lon_grid_overlap, lat_grid_overlap)

            # Interpolate and smooth data
            z_grid_overlap = self._interpolate_bathymetry(tile_data, lon_mesh_overlap, lat_mesh_overlap)
            z_smoothed = self._smooth_and_crop(z_grid_overlap, resolution_lon, resolution_lat)

            # Create plot
            fig, _ = self._create_plot(z_smoothed, min_lon, max_lon, min_lat, max_lat, lon_correction)

            # Save SVG
            svg_filename = self.output_dir / f'tile_{min_lat:.6f}_{min_lon:.6f}_{max_lat:.6f}_{max_lon:.6f}.svg'
            plt.savefig(svg_filename, format='svg', bbox_inches='tight', pad_inches=0,
                       transparent=True, facecolor='none')
            plt.close(fig)

            # Create metadata
            tile_metadata = {
                "filename": svg_filename.name,
                "coordinates": {
                    "min_lat": min_lat,
                    "max_lat": max_lat,
                    "min_lon": min_lon,
                    "max_lon": max_lon
                }
            }

            # Cleanup memory
            self._cleanup_memory(
                lon_mesh_overlap, lat_mesh_overlap, z_grid_overlap, z_smoothed,
                tile_data, lon_grid_overlap, lat_grid_overlap
            )

            return tile_metadata

        except Exception as e:
            print(f"Error creating tile {tile_x}, {tile_y}: {e}")
            return None

    def generate_all_tiles(self, resolution: Optional[int] = None, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate all tiles and return metadata."""
        if resolution is None:
            resolution = self.config.resolution
        if batch_size is None:
            batch_size = self.config.batch_size

        total_tiles = (self.tile_max_x - self.tile_min_x) * (self.tile_max_y - self.tile_min_y)
        processed = 0
        all_metadata = []

        print("Starting tile generation.")

        for tile_x in range(self.tile_min_x, self.tile_max_x):
            for tile_y in range(self.tile_min_y, self.tile_max_y):
                tile_metadata = self.create_tile_svg(tile_x, tile_y, resolution)
                if tile_metadata:
                    all_metadata.append(tile_metadata)
                processed += 1

                # Force garbage collection every batch_size tiles
                if processed % batch_size == 0:
                    gc.collect()
                    print(f"Progress: {processed}/{total_tiles} tiles ({processed/total_tiles*100:.1f}%)")

        # Save all metadata to JSON file
        metadata_file = self.output_dir / 'tiles_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        print(f"Completed all {processed} tiles")
        print(f"Metadata saved to {metadata_file}")

        return all_metadata

if __name__ == "__main__":
    config = TilerConfig(
        tile_size_deg=0.01,
        resolution=1500,
        batch_size=5,
        moothing_sigma= 6.5
    )

    tiler = BathymetryTiler(
        csv_file='bathymetry_Danube.csv',
        output_dir='bathymetry_tiles',
        config=config,
    )

    tiler.generate_all_tiles()