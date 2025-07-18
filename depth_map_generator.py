import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

csv_file_path = r"C:\Users\Andrei\Desktop\danube_map\bathymetry_Danube.csv"

lat_min_range = 45.043451
lat_max_range = 45.137734
lon_min_range = 28.987073
lon_max_range = 29.184094

df = pd.read_csv(csv_file_path)

df_filtered = df[
    (df['lat'] >= lat_min_range) & (df['lat'] <= lat_max_range) &
    (df['lon'] >= lon_min_range) & (df['lon'] <= lon_max_range) &
    (df['bathymetry'] > 0)
]

df = df_filtered

# Create a grid for interpolation using your original coordinate bounds
lat_min, lat_max = lat_min_range, lat_max_range
lon_min, lon_max = lon_min_range, lon_max_range

# Create grid within your exact coordinate range
lat_grid = np.linspace(lat_min, lat_max, 100)
lon_grid = np.linspace(lon_min, lon_max, 100)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Interpolate bathymetry data
points = np.column_stack((df['lat'].values, df['lon'].values))
values = df['bathymetry'].values

# Use griddata for interpolation
bathymetry_grid = griddata(points, values, (lat_mesh, lon_mesh), method='linear')

custom_colorscale = [
    [0.0, "rgb(198, 237, 255)"],  # very light blue (shallow)
    [0.2, "rgb(158, 202, 255)"],  # light-medium blue
    [0.4, "rgb(108, 174, 214)"],  # medium blue
    [0.6, "rgb(68, 135, 195)"],   # medium-dark blue
    [0.8, "rgb(34, 89, 164)"],    # dark blue
    [1.0, "rgb(10, 54, 130)"],    # deeper dark blue (not black)
]

# Create the contour map
fig = go.Figure(data=
    go.Contour(
        z=bathymetry_grid,
        x=lon_grid,
        y=lat_grid,
        showscale=False,
        line_smoothing=0.85,
        colorscale = custom_colorscale,
        line_width=1,
        contours=dict(
            showlabels = True,
            start=0,
            end=32,
            size=1,
            labelfont = dict(
                size = 12,
                color = 'white'
            )
        )
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)

fig.write_image("bathymetry_contour_map.svg",'svg')