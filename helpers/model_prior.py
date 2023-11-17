import numpy as np
import torch
import zuko
from torch import distributions as dist
from .utils import latlong2xy
from .geographic_data import topo_data_latlon

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