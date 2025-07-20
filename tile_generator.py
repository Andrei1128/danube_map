import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
        """Create a default SVG tile for areas with no data."""
        min_lon, max_lon, min_lat, max_lat = self.get_tile_bounds(tile_x, tile_y)

        # Create empty figure with just a light blue background
        fig = go.Figure()

        # Add a light blue rectangle to indicate water with unknown depth
        fig.add_shape(
            type="rect",
            x0=min_lon, y0=min_lat,
            x1=max_lon, y1=max_lat,
            fillcolor="lightblue",
            opacity=0.3,
            line=dict(width=0)
        )

        # Add text indicating no data
        fig.add_annotation(
            x=(min_lon + max_lon) / 2,
            y=(min_lat + max_lat) / 2,
            text="No Data",
            showarrow=False,
            font=dict(size=20, color="gray"),
            opacity=0.5
        )

        # Update layout
        fig.update_layout(
            width=800,
            height=800,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                range=[min_lon, max_lon],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[min_lat, max_lat],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            plot_bgcolor='#E6F3FF',  # Light blue background
            paper_bgcolor='white',
            showlegend=False
        )

        # Save as SVG
        filename = os.path.join(self.output_dir, f'tile_{tile_x}_{tile_y}_default.svg')
        fig.write_image(filename, format='svg', width=800, height=800)

        return filename

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

        z_grid = griddata(points, values, (lon_mesh, lat_mesh), method='linear')

        fig = go.Figure(data=go.Contour(
            x=lon_grid,
            y=lat_grid,
            z=z_grid,
            colorscale='Blues',
            showscale=False,
            line_smoothing=0.85,
            line_width=1,
            contours=dict(
                showlabels = True,
                start=0,
                end=32,
                size=1,
                labelfont = dict(
                    size = 12,
                    color = 'black'
                )
            )
        ))

        fig.update_layout(
            width=800,
            height=800,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                range=[min_lon, max_lon],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[min_lat, max_lat],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        filename = os.path.join(self.output_dir, f'tile_{tile_x}_{tile_y}.svg')
        fig.write_image(filename, format='svg', width=800, height=800)

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

    metadata = tiler.generate_all_tiles(resolution=250)