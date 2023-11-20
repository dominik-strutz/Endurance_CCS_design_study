import numpy as np
import torch
import pykonal
import xarray as xr
import socket

class Pykonal_Forward:
    def __init__(self, x, z, seismic_grid):
        self.x_min, self.x_max, self.x_N = x.min(), x.max(), x.size
        self.z_min, self.z_max, self.z_N = z.min(), z.max(), z.size      

        self.dx = (self.x_max - self.x_min) / self.x_N
        self.dz = (self.z_max - self.z_min) / self.z_N
        
        # normalising
        self.max_coordinate = np.max( [(self.x_max - self.x_min), (self.z_max - self.z_min)])
        
        self.normalising = np.array([self.max_coordinate, self.max_coordinate])
        
        self.offset = np.array([self.x_min, self.z_min])
        
        self.x_min, self.x_max = self.x_min - self.offset[0], self.x_max - self.offset[0]
        self.z_min, self.z_max = self.z_min - self.offset[1], self.z_max - self.offset[1]
        
        self.x_min, self.x_max = self.x_min / self.normalising[0], self.x_max / self.normalising[0]
        self.z_min, self.z_max = self.z_min / self.normalising[1], self.z_max / self.normalising[1]
        
        self.x_N, self.z_N = self.x_N, self.z_N
        self.dx, self.dz = self.dx / self.normalising[0], self.dz / self.normalising[1]
        
        # shift to positive
        self.x_min, self.x_max = self.x_min + 1, self.x_max + 1
        self.z_min, self.z_max = self.z_min + 1, self.z_max + 1    
            
        # add buffer
        self.x_min, self.x_max = self.x_min - 1*self.dx, self.x_max + 1*self.dx
        self.z_min, self.z_max = self.z_min - 1*self.dz, self.z_max + 1*self.dz
        
        self.x_N += 2
        self.z_N += 2
        
        self.velocity = seismic_grid.values
        # add buffer
        self.velocity = np.pad(self.velocity, (1, 1), mode='edge')
                
        # print(f'x: {self.x_min} - {self.x_max} ({self.x_N}, {self.dx})')
        # print(f'z: {self.z_min} - {self.z_max} ({self.z_N}, {self.dz})')
                
        
    def __call__(self, source_coords, receiver_coords):

        source_coords = ((source_coords-self.offset) / self.normalising) + 1
        receiver_coords = ((receiver_coords-self.offset) / self.normalising) + 1
        
        # add zero in the second entry of the last dimension
        source_coords = np.insert(source_coords, 1, 0, axis=-1)
        receiver_coords = np.insert(receiver_coords, 1, 0, axis=-1)
        
        solver = pykonal.solver.PointSourceSolver()
        solver.velocity.min_coords     = self.x_min, 0, self.z_min
        solver.velocity.node_intervals = self.dx, 1, self.dz
        solver.velocity.npts           = self.x_N, 1, self.z_N
        solver.velocity.values         = self.velocity[:, None, :]
        
        solver.src_loc = source_coords
        solver.solve()
        
        solver.traveltime.values *= self.max_coordinate
                
        return solver.traveltime.resample(receiver_coords).squeeze()


class Scipy_Lookup_Interpolation:
    def __init__(self, tt_array):
        self.tt_array = tt_array

                
    def __call__(self, offsets, source_depth, receiver_depth):
                        
        interp_in = np.stack([source_depth, receiver_depth, offsets], axis=-1) 

        interp_out = self.tt_array.interp(
            source_depth=xr.DataArray(interp_in[:, 0], dims='interp'),
            receiver_depth=xr.DataArray(interp_in[:, 1], dims='interp'),
            distance=xr.DataArray(interp_in[:, 2], dims='interp'),
            kwargs = dict(fill_value=None),
        )
        
        interp_out = interp_out.values
        interp_out = torch.from_numpy(interp_out).float()
        
        return interp_out


class Torch_Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, offsets, source_depth, receiver_depth, interpolator):
        ctx.offsets = offsets
        ctx.source_depth = source_depth
        ctx.receiver_depth = receiver_depth
        
        ctx.interpolator = interpolator
        
        return interpolator(offsets, source_depth, receiver_depth)
    
    def backward(ctx, grad_output):
        interpolator = ctx.interpolator
        
        eps = 0.1
                
        unit_vec_offsets = torch.eye(1) * eps
        unit_vec_source_depth = torch.eye(1) * eps
        unit_vec_receiver_depth = torch.eye(1) * eps
        
        d_offsets = ctx.offsets.repeat(1, 1) + unit_vec_offsets.unsqueeze(1)
        d_source_depth = ctx.source_depth.repeat(1, 1) + unit_vec_source_depth.unsqueeze(1)
        d_receiver_depth = ctx.receiver_depth.repeat(1, 1) + unit_vec_receiver_depth.unsqueeze(1)

        d_offsets = (interpolator(d_offsets, ctx.source_depth, ctx.receiver_depth) - \
            interpolator(ctx.offsets, ctx.source_depth, ctx.receiver_depth)) / eps
        d_source_depth = (interpolator(ctx.offsets, d_source_depth, ctx.receiver_depth) - \
            interpolator(ctx.offsets, ctx.source_depth, ctx.receiver_depth)) / eps
        d_receiver_depth = (interpolator(ctx.offsets, ctx.source_depth, d_receiver_depth) - \
            interpolator(ctx.offsets, ctx.source_depth, ctx.receiver_depth)) / eps
        
        d_offsets = d_offsets.moveaxis(0, -1)
        d_source_depth = d_source_depth.moveaxis(0, -1)
        d_receiver_depth = d_receiver_depth.moveaxis(0, -1)
        
        d_offsets = d_offsets * grad_output.unsqueeze(-1)
        d_source_depth = d_source_depth * grad_output.unsqueeze(-1)
        d_receiver_depth = d_receiver_depth * grad_output.unsqueeze(-1)
        
        return d_offsets, d_source_depth, d_receiver_depth, None
        
class TT_Lookup:
    def __init__(self, tt_array_filename):
        

        hostname = socket.gethostname()
        if hostname == 'stream.geos.ed.ac.uk':
            self.tt_array = xr.open_dataarray(
                tt_array_filename, engine='netcdf4', format='NETCDF4')
        else:
            self.tt_array = xr.open_dataarray(
                tt_array_filename, engine='netcdf4', format='NETCDF4')

        self.tt_array = xr.open_dataarray(
            tt_array_filename, engine='netcdf4', format='NETCDF4')
        
        self.interpolator = Scipy_Lookup_Interpolation(self.tt_array)
        self.forward = Torch_Lookup.apply
        
    def __call__(self, model, design):
        
        model_shape = model.shape
        design_shape = design.shape
        
        indices = torch.cartesian_prod(
            torch.arange(model_shape[0]),
            torch.arange(design_shape[0]),
        )
        
        model = model[indices[:, 0]]
        design = design[indices[:, 1]]
        
        
        
        offsets = np.linalg.norm((model[..., :2] - design[..., :2]), axis=-1)
        source_depth = model[..., 2]
        receiver_depth = design[..., 2]

        out = self.forward(offsets, source_depth, receiver_depth, self.interpolator).reshape(model_shape[0], design_shape[0])
        
        return out
    
import tqdm
import os
import matplotlib.pyplot as plt
import torch
import zuko
from torch import distributions as dist

from .model_prior import wells_coords_xy

from .geographic_data import topo_data_xy, E, N

hostname = socket.gethostname()

prior_mean = wells_coords_xy.mean(axis=0)

prior_bound_z_min = prior_mean[2] - 5e2 * 3
prior_bound_z_max = prior_mean[2] + 5e2 * 3

source_depth_spacing = 10
source_depths = np.arange(prior_bound_z_min, prior_bound_z_max, source_depth_spacing)

offset_spacing  =  0.2 * 1e3 # 10 km
depth_spacing = offset_spacing / 10

offset_seismic_min = 0.0
offset_seismic_max = (np.max(topo_data_xy.coords['E'].data)**2 + np.max(topo_data_xy.coords['N'].data)**2)**0.5*1.1
offset_seismic     = np.arange(offset_seismic_min, offset_seismic_max, offset_spacing)
offset_N = len(offset_seismic)
d_offset = offset_seismic[1] - offset_seismic[0]

depth_seismic_min = - 40000.0
depth_seismic_max =    2000.0
depth_seismic    = np.arange(depth_seismic_min, depth_seismic_max, depth_spacing)
depth_N = len(depth_seismic)
d_depth = depth_seismic[1] - depth_seismic[0]

# print(f'offset: {offset_seismic_min} - {offset_seismic_max} ({len(offset_seismic)}, {d_offset})')
# print(f'depth:  {depth_seismic_min} - {depth_seismic_max} ({len(depth_seismic)}, {d_depth})')

seismic_grid = xr.DataArray(
    np.moveaxis(np.linspace(depth_seismic_min, depth_seismic_max, depth_N).repeat(offset_N).reshape(depth_N, offset_N), 0, -1),
    dims=('offset', 'depth'),
    coords={
        'offset': offset_seismic,
        'depth': depth_seismic,
    }
)

elevation_grid = seismic_grid.values.copy()
seismic_grid.values = np.ones_like(seismic_grid.values)*8000.0


seismic_grid.values[
    (elevation_grid > -34.150e3)
] = 7000.0

seismic_grid.values[
    (elevation_grid > -18.870e3)
] = 7000.0

seismic_grid.values[
    (elevation_grid > -7.550e3)
] = 6450.0

seismic_grid.values[
    (elevation_grid > -2.520e3)
] = 5900.0

seismic_grid.values[
    (elevation_grid > 0)
] = 4000.0

seismic_grid.values[
    (elevation_grid > 1000)
] = 330.0


E_min, E_max = E.min(), E.max()
N_min, N_max = N.min(), N.max()

receiver_depth_spacing = 10
receiver_depths = np.arange(-1700, 700, receiver_depth_spacing)
# print(f'Number of receiver depths: {len(receiver_depths)}')

distance_spacing = 500
distances = np.arange(0, ((E_max-E_min)**2 + (N_max-N_min)**2)**0.5, distance_spacing)


if not hostname == 'TP-P14s':
    filename_tt_table = f'data/eikonal_lookup/eikonal_lookup_layered_{receiver_depth_spacing:.0f}_rdepth_{source_depth_spacing:.0f}_sdepth_{distance_spacing:.0f}_distance.nc'
else:
    filename_tt_table = f'/home/dstrutz/Downloads/ssh_cache/eikonal_lookup/eikonal_lookup_layered_{receiver_depth_spacing:.0f}_rdepth_{source_depth_spacing:.0f}_sdepth_{distance_spacing:.0f}_distance.nc'
    
# print(f'Filename: {filename_tt_table}')

if os.path.exists(filename_tt_table):
    
    if hostname not in ['stream.geos.ed.ac.uk',]:
        tt_array = xr.load_dataarray(
            filename_tt_table)
    else:
        tt_array = xr.load_dataarray(
            filename_tt_table, engine='netcdf4', format='NETCDF4')

else:
    receivers = torch.cartesian_prod(
    torch.from_numpy(distances).double(),
    torch.from_numpy(source_depths).double())
    receivers = receivers.numpy()
    
    pyk_forward = Pykonal_Forward(
        offset_seismic, depth_seismic, seismic_grid)
    
    for i, rec_depth in tqdm.tqdm(enumerate(receiver_depths), total=len(receiver_depths)):
        
        d_type = np.float32

        tt_array = xr.DataArray(
            np.zeros((len(source_depths), len(receiver_depths), len(distances)), dtype=d_type),
            dims=['source_depth', 'receiver_depth', 'distance'],
            coords=dict(
                source_depth=source_depths,
                receiver_depth=receiver_depths,
                distance=distances,
            ),
        )
        
        # print('Array dimensions: ', tt_array.shape)
        # print('Memory usage: {:.2f} GB'.format(tt_array.nbytes / 1e9))
        
        source = np.array([0, rec_depth])
        
        out = pyk_forward(
            source_coords=source,
            receiver_coords=receivers,)
        
        out = out.reshape(len(distances), len(source_depths)).T
        
        tt_array[:, i, :] = out
        
    tt_array.to_netcdf(filename_tt_table, engine='netcdf4', format='NETCDF4')

Forward_Class = TT_Lookup(filename_tt_table)
