import numpy as np
import astropy.coordinates
from astropy.coordinates import Galactic
import astropy.units as u
import astropy.time
import xarray as xr
import necstdb
from datetime import datetime

from .kisa_rev import apply_kisa_test
from .doppler import Doppler


class Initial_array(object):
    def __init__(self, path_to_data, path_to_kisa_param, xFFTS_Data_topic):

        self.topic = xFFTS_Data_topic
        self.path = path_to_data
        self.path_to_kisa_param = path_to_kisa_param

    def get_data_array(self):
        path = self.path
        topic = self.topic

        db = necstdb.opendb(path)
        xFFTS_data = db.open_table(topic).read(astype="array")
        obsmode = db.open_table("obsmode").read(astype="array")
        enc = db.open_table("status_encoder").read(astype="array")

        spec_timestamp = [datetime.utcfromtimestamp(t) for t in xFFTS_data["timestamp"]]
        obsmode_timestamp = [
            datetime.utcfromtimestamp(t) for t in obsmode["received_time"]
        ]
        enc_timestamp = [datetime.utcfromtimestamp(t) for t in enc["timestamp"]]

        data_array = xr.DataArray(
            xFFTS_data["spec"],
            dims=["t", "spectral_data"],
            coords={"t": spec_timestamp},
        )

        obsmode_array = xr.DataArray(
            obsmode["obs_mode"],
            dims=["t"],
            coords={
                "t": obsmode_timestamp,
                "scan_num": ("t", obsmode["scan_num"]),
            },
        )

        az_array = xr.DataArray(
            enc["enc_az"] / 3600, dims=["t"], coords={"t": enc_timestamp}
        )

        el_array = xr.DataArray(
            enc["enc_el"] / 3600, dims=["t"], coords={"t": enc_timestamp}
        )
        self.data_array = data_array
        self.obsmode_array = obsmode_array
        self.az_array = az_array
        self.el_array = el_array

        return data_array, obsmode_array, az_array, el_array

    def get_tp_array(self):
        path = self.path
        topic = self.topic

        db = necstdb.opendb(path)
        xFFTS_data = db.open_table(topic).read(astype="array")
        obsmode = db.open_table("obsmode").read(astype="array")
        enc = db.open_table("status_encoder").read(astype="array")

        data_array = xr.DataArray(
            xFFTS_data,
            dims=["t"],
            coords={"t": xFFTS_data["timestamp"]},
        )

        obsmode_array = xr.DataArray(
            obsmode["obs_mode"],
            dims=["t"],
            coords={
                "t": obsmode["received_time"],
                "scan_num": ("t", obsmode["scan_num"]),
            },
        )

        az_array = xr.DataArray(
            enc["enc_az"] / 3600, dims=["t"], coords={"t": enc["timestamp"]}
        )

        el_array = xr.DataArray(
            enc["enc_el"] / 3600, dims=["t"], coords={"t": enc["timestamp"]}
        )
        self.data_array = data_array
        self.obsmode_array = obsmode_array
        self.az_array = az_array
        self.el_array = el_array

        return data_array, obsmode_array, az_array, el_array

    @staticmethod
    def apply_kisa(self):

        d_az, d_el = apply_kisa_test(
            azel=(self.az_array, self.el_array), hosei=self.path_to_kisa_param
        )

        kisa_applyed_az = self.az_array + d_az / 3600
        kisa_applyed_el = self.el_array + d_el / 3600

        self.kisa_applyed_az = kisa_applyed_az
        self.kisa_applyed_el = kisa_applyed_el

        return kisa_applyed_az, kisa_applyed_el

    @staticmethod
    def apply_collimation_params(self, collimation_params):

        r = collimation_params[0]
        theta = collimation_params[1]
        d1 = collimation_params[2]
        d2 = collimation_params[3]
        El = self.kisa_applyed_el

        dAz = (r * np.cos(np.deg2rad(theta) - np.deg2rad(El)) + d1) / 3600
        dEl = (r * np.sin(np.deg2rad(theta) - np.deg2rad(El)) + d2) / 3600

        collimation_applyed_az = self.kisa_applyed_az + dAz
        collimation_applyed_el = self.kisa_applyed_el + dEl

        self.collimation_applyed_az = collimation_applyed_az
        self.collimation_applyed_el = collimation_applyed_el

        return collimation_applyed_az, collimation_applyed_el

    def concatenate(self, centre_beam=True):

        data_array = self.data_array

        obsmode_number_array = []
        for obsmode in self.obsmode_array:
            mode_number = np.nan
            if obsmode == b"HOT       ":
                mode_number = 0.0
            elif (obsmode == b"OFF       ") or (obsmode == b"SKY       "):
                mode_number = 1.0
            elif obsmode == b"ON        ":
                mode_number = 2.0
            else:
                pass
            obsmode_number_array.append(mode_number)

        self.obsmode_array_int = xr.DataArray(
            np.array(obsmode_number_array),
            dims=["t"],
            coords={"t": self.obsmode_array["t"]},
        )

        reindexed_scannum_array = self.obsmode_array.reindex(
            t=self.data_array["t"], method="pad"
        )
        reindexed_obsmode_array = self.obsmode_array_int.interp_like(self.data_array)

        if centre_beam:
            reindexed_encoder_az_array = self.kisa_applyed_az.interp_like(
                self.data_array
            )

            reindexed_encoder_el_array = self.kisa_applyed_el.interp_like(
                self.data_array
            )

        else:
            reindexed_encoder_az_array = self.collimation_applyed_az.interp_like(
                self.data_array
            )
            reindexed_encoder_el_array = self.collimation_applyed_el.interp_like(
                self.data_array
            )

        concatenated_array = data_array.assign_coords(
            obsmode=("t", reindexed_obsmode_array),
            scan_num=("t", reindexed_scannum_array["scan_num"].values),
            azlist=("t", reindexed_encoder_az_array),
            ellist=("t", reindexed_encoder_el_array),
        )

        self.concatenated_array = concatenated_array

        return concatenated_array

    @staticmethod
    def ch2velo(self, arg_dict):

        dp = Doppler(arg_dict)
        velo_list = dp.ch_speed()

        self.velo_list = velo_list
        self.concatenated_array = self.concatenated_array.assign_coords(
            vlsr=("spectral_data", velo_list)
        )
        return velo_list

    def get_lb(self):

        location = astropy.coordinates.EarthLocation(
            lon=-67.70308139 * u.deg, lat=-22.96995611 * u.deg, height=4863.85 * u.m
        )

        AltAzcoordiantes = astropy.coordinates.SkyCoord(
            az=self.concatenated_array["azlist"],
            alt=self.concatenated_array["ellist"],
            frame="altaz",
            obstime=self.concatenated_array["t"],
            location=location,
            unit="deg",
        )

        self.l_list = AltAzcoordiantes.transform_to(Galactic).l
        self.b_list = AltAzcoordiantes.transform_to(Galactic).b
        self.ra_list = AltAzcoordiantes.transform_to("fk5").ra
        self.dec_list = AltAzcoordiantes.transform_to("fk5").dec

        return self.l_list, self.b_list, self.ra_list, self.dec_list

    def make_data_array(self):

        initial_processed_array = self.concatenated_array.assign_coords(
            l_list=("t", self.l_list),
            b_list=("t", self.b_list),
            ra_list=("t", self.ra_list),
            dec_list=("t", self.dec_list),
        )

        self.initial_processed_array = initial_processed_array
        return initial_processed_array
