import matplotlib
import numpy as np 
import xarray as xr

import torch

import cartopy
from cartopy.geodesic import Geodesic

def plot_cities(ax, projPC, landmarks):
    
    ax.scatter(
        [], [],
        color='blue', marker='o', s=30, transform=projPC,
        label='Cities')
    
    for i, txt in enumerate(landmarks.index):    
        ax.scatter(
            landmarks['lon'][txt], landmarks['lat'][txt],
            color='blue', marker='o', s=30, transform=projPC,
            )

        ax.annotate(
            txt, 
            xy = (landmarks['lon'][i], landmarks['lat'][i]),
            xytext = (landmarks['lon'][i], landmarks['lat'][i]-0.04),
            size=7,
            horizontalalignment='center',
            verticalalignment='top',
            transform=projPC)

def plot_geographic_settings(
    ax, lon_min, lon_max, lat_min, lat_max,
    landmarks_df, endurance_area_latlon, hornsea_4_latlon,
    seismic_inventory, projPC,
    add_mine=None,
    add_cities=True,
    add_land=False,
    add_coastlines=False,):
    
    ax.set_extent(
    [lon_min, lon_max, lat_min, lat_max],
    crs=projPC
    )
    
    ax.fill(
        endurance_area_latlon[:, 1], endurance_area_latlon[:, 0],
        facecolor=(1,0,0,.2),
        edgecolor=(1,0,0, 1), linewidth=1.0, linestyle=':',
        transform=projPC, label='Endurance area',
    )

    # ax.plot(hornsea_4_latlon[:, 1], hornsea_4_latlon[:, 0], color='blue', linewidth=1.5, alpha=0.5, linestyle='--', transform=projPC)
    ax.fill(
        hornsea_4_latlon[:, 1], hornsea_4_latlon[:, 0],
        facecolor=(0,0,1,.2),
        edgecolor=(0,0,1, 1), linewidth=1.0, linestyle=':',
        transform=projPC, label='Hornsea 4 area',
    )

    if add_mine is not None:
        ax.scatter(
            add_mine[1], add_mine[0],
            color='k', marker='s', s=30, transform=projPC, label='Boulby mine'
            )
        
    if add_cities:
        plot_cities(ax, projPC, landmarks_df)

    ax.scatter(
        seismic_inventory['lon'], seismic_inventory['lat'], transform=projPC,
        s=40, marker='^', color='red', label='Seismic stations')


    ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, alpha=0.0)
    ax.legend(loc='upper right', frameon=True, facecolor='white')

    if add_coastlines:
        ax.coastlines()

    if add_land:
        ax.add_feature(cartopy.feature.LAND, zorder=-1, edgecolor='None')
        
    

    
            
class FixPointNormalize(matplotlib.colors.Normalize):
    ''' 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the 'sea level' 
    to a color in the blue/turquise range. 
    '''
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    

def construct3Dmodel(z, crust_1_dataset, property='vp'):
    
    x = crust_1_dataset['E'].data
    y = crust_1_dataset['N'].data
    
    property_air = np.nan
    
    if property == 'vp':
        property_air = 330.0
    elif property == 'vs':
        property_air = 0.001 # not zero to avoid division by zero
    elif property == 'rho':
        property_air = 1.293
    
    # Create a 3D model
    model = xr.DataArray(
        np.ones((len(x), len(y), len(z)))*property_air,
        dims=['x', 'y', 'z'],
        coords={'x': x, 'y': y, 'z': z},
    )
    
    layers = ['water', 'upper_sediments', 'middle_sediments', 'lower_sediments',
              'upper_crust', 'middle_crust', 'lower_crust',]
    
    for l_j in layers:
        layer_top  = l_j + '_top'
        layer_prop = l_j + '_' + property
        
        layer_top_data = crust_1_dataset[layer_top]
        
        if layer_top_data.isnull().any():
            continue
        if not (layer_top_data > z[-1]/1e3).all():
            continue
        
        for i, z_i in enumerate(z):
            mask = np.array(layer_top_data.data >= z_i/1e3) 
            model[:, :, i].data[mask] = crust_1_dataset[layer_prop].data[mask]*1e3

    return model

def construct3Dseismicmodel(z, topo_data, ):
    vp_model = xr.open_dataset('./data/CRUST1.0-vp.r0.1.nc')
    vs_model = xr.open_dataset('./data/CRUST1.0-vs.r0.1.nc')
    rho_model = xr.open_dataset('./data/CRUST1.0-rho.r0.1.nc')

    lat_min, lat_max = topo_data.coords['latitude'].min(), topo_data.coords['latitude'].max()
    lon_min, lon_max = topo_data.coords['longitude'].min(), topo_data.coords['longitude'].max()

    crust_1_dataset = xr.merge([vp_model, vs_model, rho_model])
    crust_1_dataset = crust_1_dataset.sel(latitude=slice(lat_min-2, lat_max+2), longitude=slice(lon_min-2, lon_max+2)) # some padding for interpolation

    crust_1_dataset = crust_1_dataset.interp(latitude=topo_data.coords['latitude'], longitude=topo_data.coords['longitude'], method='quadratic')
    # vp_model = vp_model.interp(latitude=topo_data.coords['latitude'], longitude=topo_data.coords['longitude'], method='nearest')

    crust_1_dataset['upper_sediments_top'] = topo_data/1e3
    crust_1_dataset['water_top'].values = np.zeros_like(crust_1_dataset['water_top'].values)


    return xr.Dataset(
        {
            'vp': construct3Dmodel(z, crust_1_dataset, property='vp'),
            'vs': construct3Dmodel(z, crust_1_dataset, property='vs'),
            'rho': construct3Dmodel(z, crust_1_dataset, property='rho'),
        }
    )

def add_distance_coordinates(data_array):
    geod = Geodesic()

    E, N = np.meshgrid(data_array.coords['lon'], data_array.coords['lat'], indexing='ij')
    N_coords = np.stack([E[0, :], N[0, :]], axis=-1)[:, ::-1]
    E_coords = np.stack([E[:, 1], N[:, 1]], axis=-1)[:, ::-1]

    del E, N

    E_km = geod.inverse(E_coords[:1], E_coords)[:, 0]
    N_km = geod.inverse(N_coords[:1], N_coords)[:, 0]

    data_array = data_array.assign_coords({'E': ('lon', E_km,), 'N': ('lat', N_km,)})
    
    return data_array


def latlong2xy(lat, lon, topo_xy, topo_latlong):
    
    lat_grid, lon_grid = topo_latlong.coords['lat'].values, topo_latlong.coords['lon'].values
    x_grid, y_grid = topo_xy.coords['E'].values, topo_xy.coords['N'].values
    
    lat_indices = []
    lon_indices = []
    
    for lat_i, lon_i in zip(lat, lon):
        lat_indices.append(np.argmin(np.abs(lat_grid - lat_i)))
        lon_indices.append(np.argmin(np.abs(lon_grid - lon_i)))
    
    return x_grid[lon_indices], y_grid[lat_indices]
    
def xy2latlong(x, y, topo_xy, topo_latlong):
        
    lat_grid, lon_grid = topo_latlong.coords['lat'].values, topo_latlong.coords['lon'].values
    x_grid, y_grid = topo_xy.coords['E'].values, topo_xy.coords['N'].values
        
    x_indices = []
    y_indices = []
    
    for x_i, y_i in zip(x, y):
        x_indices.append(np.argmin(np.abs(x_grid - x_i)))
        y_indices.append(np.argmin(np.abs(y_grid - y_i)))
    
    return lat_grid[y_indices], lon_grid[x_indices]

def plot_modelspace_dist_slice(
    ax, p_dist, XX, YY, ZZ,
    x, y, z,
    slice_axis='x', slice_value=0,
    hornsea = None, endurance = None,
    wells=None, true_event=None,
    contour_kwargs={'levels': 10, 'cmap': 'Blues', 'alpha': 0.5, 'zorder': 0},
    aspect_equal=True):
    
    local_x_min, local_x_max = x.min(), x.max()
    local_y_min, local_y_max = y.min(), y.max()
    local_z_min, local_z_max = z.min(), z.max()
    
    if slice_axis == 'z':
        z_index = torch.argmin(torch.abs(z - slice_value)).item()
        ax.contourf(
            XX[:, :, z_index], YY[:, :, z_index], p_dist[:, :, z_index].exp(),
            **contour_kwargs)

        if hornsea is not None:
            ax.plot(hornsea[:, 0], hornsea[:, 1], color='blue', linewidth=2, alpha=0.2, linestyle='--')
            ax.fill(hornsea[:, 0], hornsea[:, 1], color='blue', alpha=0.01)
            ax.plot([], [], color='blue', linewidth=2, alpha=0.5, linestyle='--', label='Hornsea 4 area')

        if endurance is not None:
            ax.plot(endurance[:, 0], endurance[:, 1], color='red', linewidth=2, alpha=0.2, linestyle='--')
            ax.fill(endurance[:, 0], endurance[:, 1], color='red', alpha=0.01,)
            ax.plot([], [], color='red', linewidth=2, alpha=0.5, linestyle='--', label='Endurance area')
            
        if true_event is not None:
            ax.scatter(
                true_event[0], true_event[1],
                marker='x', color='black', label='true event', s=50, linewidth=2)
            
        if wells is not None:
            ax.scatter(
                wells[:, 0], wells[:, 1], alpha=0.5,
                marker='.', color='black', label='Wells', s=100)

            
        ax.set_xlim((local_x_min, local_x_max))
        ax.set_ylim((local_y_min, local_y_max))

        ax.set_xticks(np.linspace(local_x_min, local_x_max, 6))
        ax.set_yticks(np.linspace(local_y_min, local_y_max, 6))

        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Northing [km]')

        ax.set_aspect('equal', 'box')

    
    if slice_axis == 'N':
        
        y_index = torch.argmin(torch.abs(y - slice_value)).item()
        ax.contourf(
            XX[:, y_index, :,], ZZ[:, y_index, :,], p_dist[:, y_index, :,].exp(),
            **contour_kwargs)

        if true_event is not None:
            ax.scatter(
                true_event[0], true_event[1],
                marker='x', color='black', label='true event', s=50, linewidth=2)
        
        ax.set_xlim((local_x_min, local_x_max))
        ax.set_ylim((local_z_min, local_z_max))

        ax.set_xticks(np.linspace(local_x_min, local_x_max, 6))
        ax.set_yticks(np.linspace(local_z_min, local_z_max, 6))

        ax.set_xlabel('Easting [km]')
        ax.set_ylabel('Depth [km]')

    if slice_axis == 'E':
        
        x_index = torch.argmin(torch.abs(x - slice_value)).item()
        ax.contourf(
            YY[x_index, :, :], ZZ[x_index, :, :], p_dist[x_index, :, :].exp(),
            **contour_kwargs)

        if true_event is not None:
            ax.scatter(
                true_event[0], true_event[1],
                marker='x', color='black', label='true event', s=50, linewidth=2)
        
        ax.set_xlim((local_y_min, local_y_max))
        ax.set_ylim((local_z_min, local_z_max))

        ax.set_xticks(np.linspace(local_y_min, local_y_max, 6))
        ax.set_yticks(np.linspace(local_z_min, local_z_max, 6))

        ax.set_xlabel('Northing [km]')
        ax.set_ylabel('Depth [km]')
        

    ax.set_xticklabels([f'{x/1000:.1f}' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{y/1000:.1f}' for y in ax.get_yticks()])
    
    if hornsea is not None or endurance is not None or true_event is not None or wells is not None:
        ax.legend(loc='upper right')
    
    if aspect_equal:
        ax.set_aspect('equal', 'box')