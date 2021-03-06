import numpy as np
import xarray as xr
import astropy.units as u
from astropy.units.quantity import Quantity
from .io import Initial_array
from tqdm import tqdm


class Array_to_map(Initial_array):
    def __init__(self, path_to_basefitted_netcdf):

        super(Initial_array, self).__init__()
        data = xr.open_dataarray(path_to_basefitted_netcdf)
        self.data = data

    def get_map_center(self):
        f = open(self.path_to_obsfile, "r")
        obs_items = f.read().split("\n")
        lambda_on = obs_items[2].split("#")[0].split("=")[1]
        beta_on = obs_items[3].split("#")[0].split("=")[1]

        self.lambda_on = float(lambda_on)
        self.beta_on = float(beta_on)

        return float(lambda_on), float(beta_on)

    def make_grid(self, grid_size, map_center, grid_number):
        if not isinstance(grid_size, Quantity):
            raise (TypeError("grid units must be specified"))
        else:
            pass

        if not isinstance(map_center, tuple):
            raise (TypeError("map center must be given as a tuple of coordinate"))

        if (not isinstance(map_center[0], Quantity)) & (
            not isinstance(map_center[1], Quantity)
        ):
            raise (TypeError("map center units must be specified"))

        else:
            pass

        grid_size_deg = grid_size.to(u.deg).value
        map_center = np.array(
            [map_center[0].to(u.deg).value, map_center[1].to(u.deg).value]
        )

        lon_coords = np.linspace(
            map_center[0] - grid_size_deg * grid_number / 2,
            map_center[0] + grid_size_deg * grid_number / 2,
        )
        lat_coords = np.linspace(
            map_center[1] - grid_size_deg * grid_number / 2,
            map_center[1] + grid_size_deg * grid_number / 2,
        )

        grid = np.meshgrid(lon_coords, lat_coords)

        self.grid = grid
        return grid

    def gridding(self, grid, grid_size):

        if not isinstance(grid_size, Quantity):
            raise (TypeError("grid units must be specified"))
        else:
            pass

        data = self.data
        grid = self.grid

        ch_length = self.data[0].shape

        lon_iterable = grid[0].flat
        lat_iterable = grid[1].flat

        gridded_list = []

        l_list = data["l_list"].values
        b_list = data["b_list"].values

        for lon, lat in tqdm(zip(lon_iterable, lat_iterable)):

            distance_from_grid_center = (
                np.sqrt((lon - l_list) ** 2 + (lat - b_list) ** 2)
                / grid_size.to(u.deg).value
            )

            mask = distance_from_grid_center <= 3
            if any(mask):
                weights = np.exp(-distance_from_grid_center[mask] ** 2)
                pix_spec_list = np.average(
                    data[mask],
                    weights=weights,
                    axis=0,
                )
            else:
                pix_spec_list = np.nan * np.zeros(ch_length)

            gridded_list.append(pix_spec_list)

        self.gridded_list = gridded_list

        return gridded_list
