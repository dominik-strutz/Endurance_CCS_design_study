# %%
import os
import contextlib

import torch
import torch.distributions as dist
import zuko

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('https://raw.githubusercontent.com/dominik-strutz/dotfiles/main/mystyle.mplstyle')

import shapely

import xarray as xr
import pandas as pd

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic


# %% [markdown]
# # Topographic Information

# %%
lon_min, lon_max = -2.0, 2.0
lat_min, lat_max = 52.0, 56.0

# %%
topo_filepath = 'data/south_west_UK_SRTM15Plus.asc'
topo_data_latlon = np.loadtxt(topo_filepath, skiprows=6)

topo_data_latlon = xr.DataArray(
    data=topo_data_latlon,
    dims=['lat', 'lon'],
    coords=dict(
        lon=np.linspace(lon_min, lon_max, topo_data_latlon.shape[1]),
        lat=np.linspace(lat_min, lat_max, topo_data_latlon.shape[0])[::-1],
    ),
    )

# %%
def add_distance_coordinates(data_array):
    geod = Geodesic()

    E, N = np.meshgrid(data_array.coords['lon'], data_array.coords['lat'], indexing='ij')
    N_coords = np.stack([E[0, :], N[0, :]], axis=-1)[:, ::-1]
    E_coords = np.stack([E[:, 1], N[:, 1]], axis=-1)[:, ::-1]

    del E, N

    E_km = geod.inverse(E_coords[:1], E_coords)[:, 0]
    N_km = geod.inverse(N_coords[:1], N_coords)[:, 0][::-1]
    
    return E_km, N_km

# %%
E, N = add_distance_coordinates(topo_data_latlon)

topo_data_xy = xr.DataArray(
    data=topo_data_latlon.data,
    dims=['N', 'E'],
    coords=dict(
        E=E,
        N=N,
    ),
    )

# %% [markdown]
# # Geographical information

# %%
from geographic_data import landmarks_df, landmarks_latlon, endurance_area_latlon, hornsea_4_latlon

def latlong2xy(lat, lon, topo_xarray):
    E, N = add_distance_coordinates(topo_xarray)
    topo_xarray = topo_xarray.assign_coords({'E': ('lon', E,), 'N': ('lat', N,)})
    topo_select = topo_xarray.interp(lon=lon, lat=lat, method='linear')
    return np.vstack((topo_select.coords['E'].data, topo_select.coords['N'].data)).T

endurance_area_xy = latlong2xy(endurance_area_latlon[:, 0], endurance_area_latlon[:, 1], topo_data_latlon)
hornsea_4_xy      = latlong2xy(hornsea_4_latlon[:, 0]     , hornsea_4_latlon[:, 1]     , topo_data_latlon)
landmarks_xy      = latlong2xy(landmarks_latlon[:, 0]     , landmarks_latlon[:, 1]     , topo_data_latlon)

# %%

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.core.inventory import read_inventory

if not os.path.exists('./data/endurance_land_stations.xml'):
    
    fdsn_client = Client('IRIS')

    #TODO: UKarray (UR) stations are not available on fdsn
    #TODO: station BEDF is missing but should be available

    inv = fdsn_client.get_stations(
        network='GB,UR',
        starttime=UTCDateTime('2023-07-01'),
        minlatitude=lat_min, maxlatitude=lat_max, 
        minlongitude=lon_min, maxlongitude=lon_max,
        level='channel'
        )

    inv.write('./data/endurance_land_stations.xml', format='stationxml')

else:
    inv = read_inventory('./data/endurance_land_stations.xml')

seismic_inventory = {}
for sta in inv[0]:
    seismic_inventory[sta.code] = {'lat': sta.latitude, 'lon': sta.longitude, 'elevation': sta.elevation}
seismic_inventory = pd.DataFrame(seismic_inventory).T

seismic_inventory['rms_noise'] = np.array([0.3,0.3,1.3,1.1,0.8,0.8,1.3,1.3,])

seismic_stations_latlon = seismic_inventory[['lat', 'lon']].values
seismic_stations_xy = latlong2xy(seismic_stations_latlon[:, 0], seismic_stations_latlon[:, 1], topo_data_latlon)

# %% [markdown]
# # Seismic Model

# %%
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

def construct3Dseismicmodel(z, topo_data, model='crust1'):
    
    
    if model == 'crust1':
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
      
    else:
        raise NotImplementedError('Only crust1 model is implemented')

# %%
# z_max_crust1 =   600 # m
# z_min_crust1 = -3600 # m

# dz = 50 #
# crust1_seimic3Dmodel = construct3Dseismicmodel(z, topo_data)


# %%
z_max_layerd =    600 # m
z_min_layerd = -40000 # m

x = topo_data_xy.coords['E'].data
y = topo_data_xy.coords['N'].data
z = np.linspace(z_min_layerd, z_max_layerd, x.shape[0])

# %% [markdown]
# # Neural Eikonal Solver

# %%
import tensorflow as tf
from tqdm.keras import TqdmCallback

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import NES

Vel = NES.velocity.HorLayeredModel(
    depths = np.array([-0.353, 2.520, 7.550, 18.870, 34.150,  50.0])*1e3,
    v     =  np.array([   4.0,   5.9,   6.45,    7.0,   8.0,   8.0])*1e3,
    xmin  =  [      0,       0, -z_max_layerd],
    xmax  =  [x.max(), y.max(), -z_min_layerd],
)

Eik_nn = NES.NES_TP(velocity=Vel, # velocity model (see NES.Interpolator)
                #  eikonal=eikonal # optional, by default isotropic eikonal equation
                 )


Eik_nn = NES.NES_TP(velocity=Vel, # velocity model (see NES.Interpolator)
                #  eikonal=eikonal # optional, by default isotropic eikonal equation
                 )

Eik_nn.build_model(
    nl=5, # number of layers
    nu=100, # number of units (may be a list)
    act='ad-gauss-1', # acivation funciton ('ad' means adaptive, '1' means slope scale)
    out_act='ad-sigmoid-1', # output activation, 'sigmoid' stands for improved factorization
    input_scale=True, # inputs scaling
    factored=True, # factorization
    out_vscale=True, # constraining by the slowest and the fastest solutions
    reciprocity=True, # symmetrizaion for the reciprocity principle 
    )

Eik_nn.compile(
    optimizer=None, # optimizer can be set manually
    loss='mae', # loss function
    lr=0.003, # learning rate for Adam optimizer
    decay=0.0005 # decay rate for Adam optimizer
    )

filepath_eikonal_nn = 'NES-TP_Model_endurance_BGS_layered'

if os.path.exists(filepath_eikonal_nn):
    Eik_nn = NES.NES_TP.load(filepath_eikonal_nn)
else:
    num_pts = 50000
    h = Eik_nn.train(
        x_train=num_pts, # number of random colocation points for training
        tolerance=2e-3, # tolerance value for early stopping (expected error with 2nd-order f-FMM)
        epochs=1000,
        verbose=0,
        callbacks=[TqdmCallback(verbose=0, miniters=10, mininterval=5)], # progress bar
        batch_size=int(num_pts/4),
    )
    
    plt.plot(h.history['loss'])
    plt.yscale('log')
    plt.show()
    
    Eik_nn.save(filepath_eikonal_nn, # path and filename which defines the folder with saved model
            save_optimizer=False, # optimizer state can be saved to continue training
            training_data=False) # training data can be saved

# %%
class Endurance_Traveltimes:
    def __init__(self, eikonal_nn_path, topo_data):
        self.topo_data = topo_data
        self.eikonal_nn_path = eikonal_nn_path
        self.eikonal_nn = None

    def _load_eikonal(self):
        if self.eikonal_nn is None:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # suppress print
                self.eikonal_nn = NES.NES_TP.load(self.eikonal_nn_path)
        else:
            pass
        
    def _discard_eikonal(self):
        del self.eikonal_nn
        self.eikonal_nn = None

    def forward(self, design, model_samples, **kwargs):
        
        self._load_eikonal()
                
        inp = self._prepare_input(design, model_samples)
        out = self.eikonal_nn.Traveltime(inp, verbose=0)
        
        out = out.reshape(model_samples.shape[0], design.shape[0])
        
        self._discard_eikonal()
        
        return torch.from_numpy(out)
    
    def hessian(self, design, model_samples, **kwargs):

        self._load_eikonal()
                
        inp = self._prepare_input(design, model_samples)
        
        out_flat = self.eikonal_nn.HessianS(inp, verbose=0)
                
        out_flat = out_flat.reshape(model_samples.shape[0], design.shape[0], model_samples.shape[-1]*2)
                
        out = np.zeros((model_samples.shape[0], design.shape[0], 3, 3))
        out[..., np.triu_indices(3)[0], np.triu_indices(3)[1]] = out_flat
        out[..., np.tril_indices(3)[0], np.tril_indices(3)[1]] = out_flat
        
        self._discard_eikonal()
        
        return torch.from_numpy(out)
    
    def jacobian(self, design, model_samples,  **kwargs):
        
        self._load_eikonal()

        inp = self._prepare_input(design, model_samples)
        out_flat = self.eikonal_nn.GradientS(inp, verbose=0)
        out = out_flat.reshape(model_samples.shape[0], design.shape[0], model_samples.shape[-1])
        
        self._discard_eikonal()
        
        return torch.from_numpy(out)
    
    def _prepare_input(self, design, model_samples, **kwargs):
        
        receivers = design[:, :3]
        
        # flip z-axis
        design[:, 2] = -design[:, 2]
        model_samples[:, 2] = -model_samples[:, 2]
        
        inp_indices = np.indices((model_samples.shape[0], receivers.shape[0])).reshape(2, -1).T
        inp = np.hstack((model_samples[inp_indices[:,0]], receivers[inp_indices[:,1]]))
        
        return inp

# %% [markdown]
# # Prior Distribution

# %%
wells_coords_latlon = np.array(
    [[54.20020528, 0.9916635 , -1020.0],
     [54.24513067, 0.97697506, -1020.0],
     [54.23250939, 1.03511106, -1020.0],
     [54.19777192, 1.05146139, -1020.0],
     [54.21851681, 0.96272839, -1020.0],]
)

wells_coords_xy = np.hstack( [latlong2xy(wells_coords_latlon[:, 0], wells_coords_latlon[:, 1], topo_data_latlon), wells_coords_latlon[:, 2].reshape(-1, 1)])

prior_weights    = torch.ones(wells_coords_xy.shape[0]).float()
prior_means      = torch.from_numpy(wells_coords_xy).float()
prior_covariance = torch.eye(prior_means.shape[1]).expand(prior_means.shape[0], -1, -1) * torch.tensor([5e3, 5e3, 5e2])**2

prior_dist = zuko.distributions.Mixture(dist.MultivariateNormal(
    loc=prior_means, covariance_matrix=prior_covariance), prior_weights)

prior_samples_test = prior_dist.sample((10000,))


