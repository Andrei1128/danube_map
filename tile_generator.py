import gc
import json
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from scipy.spatial import cKDTree

@dataclass
class TilerConfig:
    """Configuration for the bathymetry tiler."""
    tile_size_deg: float = 0.01
    overlap_factor: float = 0.1
    resolution: int = 1000
    batch_size: int = 10
    n_workers: int = None  # None = auto-detect CPU count
    smoothing_sigma: float = 5
    contour_max_depth: float = 40.0
    contour_step: float = 1.0
    fig_width: float = 8.0
    fig_dpi: int = 100
    tile_opacity: float = 0.9
    contour_opacity: float = 0.9
    enable_contours: bool = True
    show_depth_labels: bool = True

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
        self._data = None
        self._bounds = None
        self._spatial_index = None

    def _load_data(self) -> pd.DataFrame:
        """Load data once and cache it."""
        if self._data is None:
            self._data = pd.read_csv(self.csv_file)
            # Build spatial index for faster queries
            self._build_spatial_index()
        return self._data

    def _build_spatial_index(self) -> None:
        """Build spatial index for faster tile queries."""
        if self._data is not None and self._spatial_index is None:
            points = self._data[['lon', 'lat']].values
            self._spatial_index = cKDTree(points)

    def calculate_bounds(self) -> DataBounds:
        """Calculate data bounds and statistics."""
        if self._bounds is None:
            data = self._load_data()

            self._bounds = DataBounds(
                min_lat=data['lat'].min(),
                max_lat=data['lat'].max(),
                min_lon=data['lon'].min(),
                max_lon=data['lon'].max(),
                total_points=len(data),
                water_points=len(data)
            )

        return self._bounds

    def load_tile_data(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> pd.DataFrame:
        """Load data for a specific tile area."""
        data = self._load_data()

        # For large datasets, use spatial index for faster queries
        if len(data) > 100000 and self._spatial_index is not None:
            # Use spatial index to find points in bounding box
            # Create corners of bounding box
            bbox_corners = np.array([
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat]
            ])

            # Find all points within a reasonable distance of bbox
            center = np.array([(min_lon + max_lon) / 2, (min_lat + max_lat) / 2])
            diagonal = np.sqrt((max_lon - min_lon)**2 + (max_lat - min_lat)**2)

            # Query spatial index
            indices = self._spatial_index.query_ball_point(center, diagonal)
            candidate_data = data.iloc[indices]

            # Apply exact bounding box filter on candidates
            mask = (
                (candidate_data['lon'] >= min_lon) & (candidate_data['lon'] < max_lon) &
                (candidate_data['lat'] >= min_lat) & (candidate_data['lat'] < max_lat)
            )

            return candidate_data[mask].copy()
        else:
            # For smaller datasets, use direct filtering
            mask = (
                (data['lon'] >= min_lon) & (data['lon'] < max_lon) &
                (data['lat'] >= min_lat) & (data['lat'] < max_lat)
            )

            return data[mask].copy()

class BathymetryTiler:
    """Generates SVG tiles from bathymetry data."""

    def __init__(self, csv_file: str | Path, output_dir: str | Path = 'tiles', config: Optional[TilerConfig] = None,
                 min_lat: Optional[float] = None, max_lat: Optional[float] = None,
                 min_lon: Optional[float] = None, max_lon: Optional[float] = None):
        self.config = config or TilerConfig()
        self.output_dir = Path(output_dir)
        self.data_loader = DataLoader(csv_file)

        self.output_dir.mkdir(exist_ok=True)

        self.bounds = self.data_loader.calculate_bounds()

        # Use custom bounds if provided, otherwise use data bounds
        if min_lat is not None:
            self.bounds.min_lat = min_lat

        if max_lat is not None:
            self.bounds.max_lat = max_lat

        if min_lon is not None:
            self.bounds.min_lon = min_lon

        if max_lon is not None:
            self.bounds.max_lon = max_lon

        self.tile_min_x = floor(self.bounds.min_lon / self.config.tile_size_deg)
        self.tile_max_x = ceil(self.bounds.max_lon / self.config.tile_size_deg)
        self.tile_min_y = floor(self.bounds.min_lat / self.config.tile_size_deg)
        self.tile_max_y = ceil(self.bounds.max_lat / self.config.tile_size_deg)

        self.n_tiles_x = self.tile_max_x - self.tile_min_x
        self.n_tiles_y = self.tile_max_y - self.tile_min_y

        # Set number of workers
        if self.config.n_workers is None:
            self.config.n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues

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

        # Remove any NaN or infinite values from input data
        valid_indices = np.isfinite(values) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
        points = points[valid_indices]
        values = values[valid_indices]

        if len(points) < 4:
            raise ValueError("Not enough valid points after filtering NaN/infinite values")

        z_grid = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

        # Replace NaN values with 0 (water surface)
        z_grid = np.nan_to_num(z_grid, nan=0.0, posinf=0.0, neginf=0.0)

        # Cleanup input arrays
        del points, values
        return z_grid

    def _smooth_and_crop(self, z_grid: np.ndarray, resolution_lon: int, resolution_lat: int) -> np.ndarray:
        """Apply smoothing and crop to exact tile boundaries."""
        z_smoothed = gaussian_filter(z_grid, sigma=self.config.smoothing_sigma)

        # Crop to exact tile boundaries
        overlap_pixels_lon = int(resolution_lon * self.config.overlap_factor)
        overlap_pixels_lat = int(resolution_lat * self.config.overlap_factor)

        core_start_x = overlap_pixels_lon
        core_end_x = z_smoothed.shape[1] - overlap_pixels_lon
        core_start_y = overlap_pixels_lat
        core_end_y = z_smoothed.shape[0] - overlap_pixels_lat

        return z_smoothed[core_start_y:core_end_y, core_start_x:core_end_x]

    def _create_plot(self, z_smoothed: np.ndarray, min_lon: float, max_lon: float, min_lat: float, max_lat: float, lon_correction: float) -> plt.Figure:
        """Create the matplotlib plot for the tile."""
        fig_height = self.config.fig_width / lon_correction
        fig, ax = plt.subplots(figsize=(self.config.fig_width, fig_height), dpi=self.config.fig_dpi)

        # Create coordinate grids more efficiently
        lon_grid = np.linspace(min_lon, max_lon, z_smoothed.shape[1])
        lat_grid = np.linspace(min_lat, max_lat, z_smoothed.shape[0])
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid, indexing='xy')

        # Create contour plots
        contour_levels = np.arange(0, self.config.contour_max_depth, self.config.contour_step)

        # Ensure we have valid contour levels
        if len(contour_levels) == 0:
            contour_levels = np.array([0, 10, 20, 30, 40])

        # Always create filled contours for the base bathymetry visualization
        ax.contourf(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                   cmap='Blues', alpha=self.config.tile_opacity, corner_mask=False, extend='max')

        # Conditionally create contour lines and labels
        if self.config.enable_contours:
            cs_lines = ax.contour(lon_mesh, lat_mesh, z_smoothed, levels=contour_levels,
                                 colors='black', linewidths=0.9, alpha=self.config.contour_opacity,
                                 corner_mask=False, antialiased=True)

            # Only add labels if we have contour lines and depth labels are enabled
            if self.config.show_depth_labels:
                try:
                    if hasattr(cs_lines, 'collections') and len(cs_lines.collections) > 0:
                        ax.clabel(cs_lines, inline=True, fontsize=12, fmt='%d', colors='black')
                    elif hasattr(cs_lines, 'levels') and len(cs_lines.levels) > 0:
                        ax.clabel(cs_lines, inline=True, fontsize=12, fmt='%d', colors='black')
                except (AttributeError, ValueError):
                    # Skip labeling if there's an issue with contour lines
                    pass

        # Configure plot appearance
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_aspect(1.0/lon_correction)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        # Set transparent background
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Use subplots_adjust instead of tight_layout to avoid warnings
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Cleanup coordinate grids
        del lon_grid, lat_grid, lon_mesh, lat_mesh

        return fig

    def _cleanup_memory(self, *objects) -> None:
        """Clean up memory by deleting objects and running garbage collection."""
        for obj in objects:
            if obj is not None:
                del obj
        plt.close('all')  # More thorough cleanup
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
            fig = self._create_plot(z_smoothed, min_lon, max_lon, min_lat, max_lat, lon_correction)

            # Save SVG
            self.output_dir.mkdir(exist_ok=True)  # Ensure directory exists
            svg_filename = self.output_dir / f'tile_{min_lat:.6f}_{min_lon:.6f}_{max_lat:.6f}_{max_lon:.6f}.svg'
            fig.savefig(svg_filename, format='svg', bbox_inches=None, pad_inches=0,
                       transparent=True, facecolor='none', edgecolor='none')
            plt.close(fig)
            del fig

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

            # Cleanup memory more aggressively
            self._cleanup_memory(
                lon_mesh_overlap, lat_mesh_overlap, z_grid_overlap, z_smoothed,
                tile_data, lon_grid_overlap, lat_grid_overlap
            )

            return tile_metadata

        except Exception as e:
            print(f"Error creating tile ({tile_x}, {tile_y}): {e}")
            return None

    def generate_all_tiles(self, resolution: Optional[int] = None, batch_size: Optional[int] = None,
                          progress_callback: Optional[callable] = None,
                          should_stop_callback: Optional[callable] = None,
                          use_parallel: bool = True) -> List[Dict[str, Any]]:
        """Generate all tiles and return metadata."""
        if resolution is None:
            resolution = self.config.resolution
        if batch_size is None:
            batch_size = self.config.batch_size

        total_tiles = (self.tile_max_x - self.tile_min_x) * (self.tile_max_y - self.tile_min_y)
        all_metadata = []

        if progress_callback:
            progress_callback(0, total_tiles, "Starting tile generation...")

        # Generate list of all tile coordinates
        tile_coords = [(x, y) for x in range(self.tile_min_x, self.tile_max_x)
                      for y in range(self.tile_min_y, self.tile_max_y)]

        if use_parallel and self.config.n_workers > 1:
            # Parallel generation
            all_metadata = self._generate_tiles_parallel(tile_coords, resolution,
                                                        progress_callback, should_stop_callback)
        else:
            # Sequential generation (original behavior)
            processed = 0
            for tile_x, tile_y in tile_coords:
                if should_stop_callback and should_stop_callback():
                    break

                tile_metadata = self.create_tile_svg(tile_x, tile_y, resolution)
                if tile_metadata:
                    all_metadata.append(tile_metadata)
                processed += 1

                if progress_callback:
                    progress_callback(processed, total_tiles, f"Generated tile {processed}/{total_tiles}")

                if processed % batch_size == 0:
                    gc.collect()

        # Save all metadata to JSON file
        self.output_dir.mkdir(exist_ok=True)  # Ensure directory exists
        metadata_file = self.output_dir / 'tiles_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)

        if progress_callback:
            progress_callback(total_tiles, total_tiles, f"Completed all {len(all_metadata)} tiles")

        return all_metadata

    def _generate_tiles_parallel(self, tile_coords: List[Tuple[int, int]], resolution: int,
                               progress_callback: Optional[callable] = None,
                               should_stop_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Generate tiles in parallel using ThreadPoolExecutor."""
        all_metadata = []
        total_tiles = len(tile_coords)
        processed = 0

        # Create batches to avoid memory issues
        batch_size = min(self.config.batch_size, len(tile_coords))
        batches = [tile_coords[i:i + batch_size] for i in range(0, len(tile_coords), batch_size)]

        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            for batch in batches:
                if should_stop_callback and should_stop_callback():
                    break

                # Submit batch for processing
                future_to_coord = {executor.submit(self._create_tile_worker, coord[0], coord[1], resolution): coord
                                 for coord in batch}

                # Collect results
                for future in as_completed(future_to_coord):
                    if should_stop_callback and should_stop_callback():
                        break

                    try:
                        tile_metadata = future.result()
                        if tile_metadata:
                            all_metadata.append(tile_metadata)
                    except Exception as e:
                        coord = future_to_coord[future]
                        print(f"Error processing tile {coord}: {e}")

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total_tiles, f"Generated tile {processed}/{total_tiles}")

                # Force garbage collection after each batch
                gc.collect()

        return all_metadata

    def _create_tile_worker(self, tile_x: int, tile_y: int, resolution: int) -> Optional[Dict[str, Any]]:
        """Worker function for parallel tile generation."""
        return self.create_tile_svg(tile_x, tile_y, resolution)