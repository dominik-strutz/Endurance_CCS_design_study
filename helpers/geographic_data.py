import os

import numpy as np
import pandas as pd
import xarray as xr

from .utils import *

landmarks = {
    'Hull': {'lat': 53.745670, 'lon': -0.336741},
    'Scarbrough': {'lat': 54.283113, 'lon': -0.399752},
    # 'Whitby': {'lat': 54.486335, 'lon': -0.613347},
    # 'York': {'lat': 53.959965, 'lon': -1.087298},
    'Leeds': {'lat': 53.800755, 'lon': -1.549077},
    'Sheffield': {'lat': 53.381129, 'lon': -1.470085},
    'Grimsby': {'lat': 53.565269, 'lon': -0.075683},
    # 'Scunthorpe': {'lat': 53.588646, 'lon': -0.654413},
    'Middlesbrough': {'lat': 54.574227, 'lon': -1.234956},
    # 'Redcar': {'lat': 54.609688, 'lon': -1.076133},
}

landmarks_df = pd.DataFrame(landmarks).T
landmarks_latlon = landmarks_df[['lat', 'lon']].values

# bp_ccs_area_1_latlon = np.array(
#     [[54.26666667, 1.96666667], [54.26666667, 2.16666667],[54.23333333, 2.16666667], [54.23333333, 2.23333333],
#      [54.2       , 2.23333333], [54.2       , 2.3       ],[54.16666667, 2.3       ], [54.16666667, 2.36666667],
#      [54.03333333, 2.36666667], [54.03333333, 2.26666667],[54.        , 2.26666667], [54.        , 1.9       ],
#      [54.08333333, 1.9       ], [54.08333333, 1.95      ],[54.1       , 1.95      ], [54.1       , 1.96666667],
#      [54.26666667, 1.96666667],]
# )

# bp_ccs_area_2_latlon = np.array(
#     [[54.33333333, 1.4       ], [54.33333333, 1.6       ], [54.3       , 1.6       ], [54.3       , 1.83333333],
#      [54.26666667, 1.83333333], [54.26666667, 1.96666667], [54.1       , 1.96666667], [54.1       , 1.95      ],
#      [54.08333333, 1.95      ], [54.08333333, 1.9       ], [54.06666667, 1.9       ], [54.06666667, 1.76666667],
#      [54.1       , 1.76666667], [54.1       , 1.7       ], [54.13333333, 1.7       ], [54.13333333, 1.63333333],
#      [54.16666667, 1.63333333], [54.16666667, 1.4       ], [54.33333333, 1.4       ],]
# )

endurance_area_latlon = np.array(
    [[54.28580772,  0.83107564],[54.28635239,  1.05000011], [54.26684075,  1.04981736], [54.26688633,  1.06650142], 
     [54.24909494,  1.06666664],[54.24927728,  1.23243322], [54.13342356,  1.23348467], [54.133192  ,  0.93304544], 
     [54.16671497,  0.93301289],[54.1667035 ,  0.87656772], [54.20238397,  0.87702906], [54.20274228,  0.83110422],
     [54.28580772,  0.83107564]]
)

hornsea_4_latlon = np.array(
    [
     [54.04757, 1.13835], [54.12361, 0.99853], [54.15375, 1.01218], [54.18041, 0.97272], 
     [54.21032, 0.97530], [54.20484, 1.20507], [54.07028, 1.50146], [53.98767, 1.28907],
     [54.00648, 1.21356], [54.04757, 1.13835]
    ]
)


york_array_centre_latlon = np.array(
    [54.386798, -0.663078]
)

boulby_mine_latlon = np.array(
    [54.550460,  -0.819456, -1500]
)

LON_MIN, LON_MAX = -2.0, 2.0
LAT_MIN, LAT_MAX = 52.0, 56.0

topo_filepath = 'data/south_west_UK_SRTM15Plus.asc'
topo_data_latlon = np.loadtxt(topo_filepath, skiprows=6)

topo_data_latlon = xr.DataArray(
    data=topo_data_latlon,
    dims=['lat', 'lon'],
    coords=dict(
        lon=np.linspace(LON_MIN, LON_MAX, topo_data_latlon.shape[1]),
        lat=np.linspace(LAT_MIN, LAT_MAX, topo_data_latlon.shape[0])[::-1],
    ),
    )

E, N = add_distance_coordinates(topo_data_latlon)

topo_data_xy = xr.DataArray(
    data=topo_data_latlon.data,
    dims=['N', 'E'],
    coords=dict(
        E=E,
        N=N,
    ),
    )

endurance_area_xy = latlong2xy(endurance_area_latlon[:, 0], endurance_area_latlon[:, 1], topo_data_latlon)
hornsea_4_xy      = latlong2xy(hornsea_4_latlon[:, 0]     , hornsea_4_latlon[:, 1]     , topo_data_latlon)
landmarks_xy      = latlong2xy(landmarks_latlon[:, 0]     , landmarks_latlon[:, 1]     , topo_data_latlon)

if not os.path.exists('data/endurance_land_stations.pkl'):
    
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.core.inventory import read_inventory
    
    fdsn_client = Client('IRIS')

    #TODO: UKarray (UR) stations are not available on fdsn
    #TODO: station BEDF is missing but should be available

    inv = fdsn_client.get_stations(
        network='GB,UR',
        starttime=UTCDateTime('2023-07-01'),
        minlatitude=LAT_MIN, maxlatitude=LAT_MAX, 
        minlongitude=LON_MIN, maxlongitude=LON_MAX,
        level='channel'
        )
    
    seismic_inventory = {}
    for sta in inv[0]:
        seismic_inventory[sta.code] = {'lat': sta.latitude, 'lon': sta.longitude, 'elevation': sta.elevation}
    seismic_inventory = pd.DataFrame(seismic_inventory).T
    
    seismic_inventory.to_pickle('./data/endurance_land_stations.pkl')
else:
    seismic_inventory = pd.read_pickle('./data/endurance_land_stations.pkl') 

seismic_inventory['rms_noise'] = np.array([0.3,0.3,1.3,1.1,0.8,0.8,1.3,1.3,])

seismic_stations_latlon = seismic_inventory[['lat', 'lon']].values
seismic_stations_xy = latlong2xy(seismic_stations_latlon[:, 0], seismic_stations_latlon[:, 1], topo_data_latlon)