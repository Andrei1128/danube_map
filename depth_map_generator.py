import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import plotly.express as px

csv_file_path = r"C:\Users\Andrei\Desktop\danube_map\bathymetry_Danube.csv"

lat_min_range = 45.043451
lat_max_range = 45.137734
lon_min_range = 28.987073
lon_max_range = 29.184094

df = pd.read_csv(csv_file_path)

df_filtered = df[
    (df['lat'] >= lat_min_range) & (df['lat'] <= lat_max_range) &
    (df['lon'] >= lon_min_range) & (df['lon'] <= lon_max_range) &
    df['bathymetry'] > 0
]

df = df_filtered

# Create a grid for interpolation using your original coordinate bounds
lat_min, lat_max = lat_min_range, lat_max_range
lon_min, lon_max = lon_min_range, lon_max_range

# Create grid within your exact coordinate range
lat_grid = np.linspace(lat_min, lat_max, 800)
lon_grid = np.linspace(lon_min, lon_max, 800)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Interpolate bathymetry data
points = np.column_stack((df['lat'].values, df['lon'].values))
values = df['bathymetry'].values

# Use griddata for interpolation
bathymetry_grid = griddata(points, values, (lat_mesh, lon_mesh), method='linear')

# Create the contour map
fig = go.Figure(data=
    go.Contour(
        z=bathymetry_grid,
        x=lon_grid,
        y=lat_grid,
        showscale=False,
        line_smoothing=0.85,
        contours=dict(
            showlabels = True,
            labelfont = dict(
                size = 12,
                color = 'white',
            )
        )
))

# Save as HTML for easy viewing
fig.write_image("bathymetry_contour_map.svg",'svg')