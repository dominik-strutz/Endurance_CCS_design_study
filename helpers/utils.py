import numpy as np
from cartopy.geodesic import Geodesic
import xarray as xr
import torch
import torch.distributions as dist

def add_distance_coordinates(data_array):
    geod = Geodesic()

    E, N = np.meshgrid(data_array.coords['lon'], data_array.coords['lat'], indexing='ij')
    N_coords = np.stack([E[0, :], N[0, :]], axis=-1)[:, ::-1]
    E_coords = np.stack([E[:, 1], N[:, 1]], axis=-1)[:, ::-1]

    del E, N

    E_km = geod.inverse(E_coords[:1], E_coords)[:, 0]
    N_km = geod.inverse(N_coords[:1], N_coords)[:, 0][::-1]
    
    return E_km, N_km

def latlong2xy(lat, lon, topo_xarray):
    E, N = add_distance_coordinates(topo_xarray)
    topo_xarray = topo_xarray.assign_coords({'E': ('lon', E,), 'N': ('lat', N,)})
    topo_select = topo_xarray.interp(lon=lon, lat=lat, method='linear')
    return np.vstack((topo_select.coords['E'].data, topo_select.coords['N'].data)).T

def get_elevation(points, topo_ds):
    east = xr.DataArray(points[..., 0], dims='points')
    north = xr.DataArray(points[..., 1], dims='points')
    elevations = topo_ds.interp(
        E=east, N=north, method='nearest').values
    return torch.from_numpy(elevations)

def gridsearch_posterior(tt_obs, x, y, z, design, prior_dist, data_likelihood):
    p_posterior_X, p_posterior_Y, p_posterior_Z = torch.meshgrid(x, y, z, indexing='ij')
    posterior_grid = torch.stack([p_posterior_X, p_posterior_Y, p_posterior_Z], axis=-1)

    p_likelihood = data_likelihood(posterior_grid, design).log_prob(tt_obs)
    p_prior = prior_dist.log_prob(posterior_grid)

    p_unnormalised_posterior = p_likelihood + p_prior
    p_evidence = torch.logsumexp(p_unnormalised_posterior, dim=(0,1,2))

    p_posterior = p_unnormalised_posterior - p_evidence
    
    return p_posterior, p_prior, p_posterior_X, p_posterior_Y, p_posterior_Z


class Data_Likelihood:
    def __init__(
        self, Forward_Class,
        const_noise_term_multiplier = 1.0,
        quad_noise_term = 0.02
        ):
        
        self.Forward_Class = Forward_Class
        self.const_noise_term_multiplier = const_noise_term_multiplier
        self.quad_noise_term = quad_noise_term
    
    def __call__(self, model_samples, design=None):
    
        model_samples_batch_shape = model_samples.shape[:-1]
        
        model_samples = model_samples.flatten(end_dim=-2)
        data_samples = self.Forward_Class(model_samples, design)
        data_samples = data_samples.reshape(model_samples_batch_shape + (-1,))
        
        std = (design[:, -1] * self.const_noise_term_multiplier) + \
            (data_samples*self.quad_noise_term)**2

        return dist.Independent(dist.Normal(data_samples, std), 1)