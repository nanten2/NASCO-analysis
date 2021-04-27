from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Union, Optional, List, Tuple, Dict, Any

import necstdb
import numpy as np
import xarray as xr
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz, FK5, Galactic
import n_const.constants as n2const

from .kisa_rev import apply_kisa_test
from .doppler import Doppler

PathLike = Union[str, Path]
timestamp2datetime = np.vectorize(datetime.utcfromtimestamp)
AC240_TP_FIELD = "POWER_BE1"


@dataclass
class InitialArray(object):
    """Read raw data and make DataArray.
    Parameters
    ----------
    data_path: str or Path
        Path to data directory where raw data (.header and .data files)
        are saved.
    kisa_path: str or Path
        Path to *kisa* file.
    topic_name: str
        Name of ROS topic through which the data taken with specific
        spectrometer board is sent.

    Notes
    -----
    ROS topic for each board is named as follows:
    - Spectral data taken by XFFTS board xx : "xffts_boardxx" (0-padded)
    - Total power data taken by XFFTS board xx: "xffts_power_boardxx" (0-padded)
    - Spectral data taken by AC240 board x: "ac240_spactra_data_x"
    - Total power data taken by AC240 board x: "ac240_tp_data_x"

    Examples
    --------
    >>> data_path = "/path/to/data/directory"
    >>> kisa_path = "path/to/kisafile.txt"
    >>> ia = InitialArray(data_path, "xffts_board01", kisa_path)
    """

    data_path: PathLike
    topic_name: str
    kisa_path: Optional[PathLike] = None

    def create_data_array(self) -> xr.Dataset:
        """Get spectral data."""

        # read data
        db = necstdb.opendb(self.data_path)
        data = db.open_table(self.topic_name).read(astype="array")
        obsmode = db.open_table("obsmode").read(astype="array")
        encoder = db.open_table("status_encoder").read(astype="array")
        weather = db.open_table("status_weather").read(astype="array")

        # convert structured array into dict of DataArrays
        def convert(
            structured_array: np.ndarray, dims: List[str]
        ) -> Dict[str, xr.DataArray]:
            if not structured_array.dtype.names:  # when not structured
                return {"total_power": xr.DataArray(structured_array, dims=dims)}
            ret = {}
            for field in structured_array.dtype.names:
                dim = dims[: structured_array[field].ndim]
                ret[field] = xr.DataArray(structured_array[field], dims=dim)
            return ret

        data = convert(data, ["t", "spectra"])
        obsmode = convert(obsmode, ["t"])
        encoder = convert(encoder, ["t"])
        weather = convert(weather, ["t"])

        # change time format
        data["timestamp"] = timestamp2datetime(data["timestamp"].astype(float))
        obsmode["received_time"] = timestamp2datetime(
            obsmode["received_time"].astype(float)
        )  # no timestamp recorded
        encoder["timestamp"] = timestamp2datetime(encoder["timestamp"].astype(float))
        weather["timestamp"] = timestamp2datetime(weather["timestamp"].astype(float))

        # create Dataset
        self.data_set = (
            xr.Dataset(data).set_index({"t": "timestamp"}).drop_vars("received_time")
        )
        self.obsmode_set = xr.Dataset(obsmode).set_index({"t": "received_time"})
        self.encoder_set = (
            xr.Dataset(encoder).set_index({"t": "timestamp"}).drop_vars("received_time")
        ) / 3600
        self.weather_set = (
            xr.Dataset(weather).set_index({"t": "timestamp"}).drop_vars("received_time")
        )

        return self.data_set

    def create_tp_array(self) -> xr.Dataset:
        """
        Notes
        -----
        This function may calculate total power from spectral data in
        the future.
        """
        data_set = self.create_data_array()
        if AC240_TP_FIELD in data_set.keys():
            not_necessary = set(data_set.keys())
            not_necessary.discard(AC240_TP_FIELD)
            self.data_set = data_set.drop_vars(not_necessary).rename_vars(
                {AC240_TP_FIELD: "total_power"}
            )
        return self.data_set

    def correct_kisa(self) -> Tuple[xr.DataArray]:
        if self.kisa_path is None:
            self.encoder_set = self.encoder_set.rename(
                {"enc_az": "az", "enc_el": "el"}
            ).assign_attrs({"kisa_applied": False})
            return
        enc_az_array = self.encoder_set.enc_az
        enc_el_array = self.encoder_set.enc_el

        dAz, dEl = apply_kisa_test(
            azel=(enc_az_array, enc_el_array), hosei=self.kisa_path
        )

        az = enc_az_array + dAz / 3600
        el = enc_el_array + dEl / 3600

        self.encoder_set = self.encoder_set.assign(
            az=("t", az), el=("t", el)
        ).assign_attrs({"kisa_applied": True})

        return az, el

    def correct_collimation_error(
        self, collimation_params: Dict[str, float] = None
    ) -> Tuple[xr.DataArray]:
        self.correct_kisa()
        if collimation_params is None:
            self.encoder_set = self.encoder_set.assign_attrs(
                {"collimation_applied": False}
            )
            return

        # specify the beam by self.topic_name using newly defined Constants?

        r = collimation_params.r  # arcsec
        theta = np.deg2rad(collimation_params.theta)
        d1 = collimation_params.d1  # arcsec
        d2 = collimation_params.d2  # arcsec
        el = np.deg2rad(self.data.el)

        dAz = (r * np.cos(theta - el) + d1) / 3600  # deg
        dEl = (r * np.sin(theta - el) + d2) / 3600  # deg

        az = self.encoder_set.az + dAz
        el = self.encoder_set.el + dEl

        self.encoder_set = self.encoder_set.assign(
            az=("t", az), el=("t", el)
        ).assign_attrs({"collimation_applied": True})
        return

    def combine_metadata(self, int_obsmode: bool = False) -> xr.DataArray:
        self.correct_kisa()

        if int_obsmode:

            @np.vectorize  # make following function accept array-like as input
            def convert(mode: bytes) -> float:
                if mode == b"HOT       ":
                    return 0.0
                elif (mode == b"OFF       ") | (mode == b"SKY       "):
                    return 1.0
                elif mode == b"ON        ":
                    return 2.0
                else:
                    return np.nan

            int_obsmode_array = convert(self.obsmode_set.obs_mode)
            self.obsmode_set = self.obsmode_set.assign(
                int_obsmode=("t", int_obsmode_array)
            )

        reindexed_obsmode_set = self.obsmode_set.reindex_like(
            self.data_set, method="ffill"
        )  # compare forward-fill and backward-fill and pad conflicting with nonsense
        reindexed_obsmode_set = reindexed_obsmode_set.where(
            reindexed_obsmode_set
            == self.obsmode_set.reindex_like(self.data_set, method="bfill"),
            # b"None",
        )
        reindexed_encoder_set = self.encoder_set.interp_like(
            self.data_set, method="linear"
        )
        reindexed_weather_set = self.weather_set.interp_like(
            self.data_set, method="linear"
        )

        # determine field name of spectral or total power data
        if "spec" in self.data_set.keys():
            main_data_field = "spec"
        elif "total_power" in self.data_set.keys():
            main_data_field = "total_power"
        else:
            raise ValueError("Spectral or total power data field not found.")

        coords = {
            "obs_mode": ("t", reindexed_obsmode_set.obs_mode),
            "scan_num": ("t", reindexed_obsmode_set.scan_num),
            "az": ("t", reindexed_encoder_set.az),
            "el": ("t", reindexed_encoder_set.el),
        }
        for key in reindexed_weather_set.keys():
            coords[key] = ("t", reindexed_weather_set[key])

        self.data = (
            self.data_set[main_data_field]
            .assign_coords(coords)
            .assign_attrs(self.encoder_set.attrs)
        )
        return self.data

    def channel2velocity(self, args: Optional[dict] = None, **kwargs) -> xr.DataArray:
        """
        Parameters
        ----------
        *args: dict
        **kwargs: Any
            Parameters ['spectrometer', 'rest_freq', 'LO1st_freq',
            'LO1st_factor', 'LO2nd_freq'] should be given as dict args
            or kwargs.
        """
        if "spec" not in self.data_set.keys():
            raise ValueError(
                "Spectral data not found."
                "Cannot calculate doppler velocity on total power data."
            )

        kwargs.update(
            {
                "obstime": self.data.t,
                "ra": self.data.ra.data * u.deg,
                "dec": self.data.dec.data * u.deg,
            }
        )
        dp = Doppler(args=args, **kwargs)
        velocity = dp.ch_speed()

        self.data = self.data.assign_coords(v_lsr=(["t", "spectra"], velocity))
        return self.data

    def convert_coordinates(self) -> xr.DataArray:
        horizontal_coord = SkyCoord(
            az=self.data.az.data * u.deg,
            alt=self.data.el.data * u.deg,
            frame=AltAz,
            obstime=self.data.t,
            location=n2const.LOC_NANTEN2,
            pressure=self.data.press.data * u.hPa,
            temperature=self.data.out_temp.data * u.deg_C,
            relative_humidity=self.data.out_humi.data * u.percent,
            # refraction have almost no dependency on wavelength at RADIO frequency
        )
        equatorial_coord = horizontal_coord.transform_to(FK5)
        galactic_coord = equatorial_coord.transform_to(Galactic)

        self.data = self.data.assign_coords(
            ra=("t", equatorial_coord.ra),
            dec=("t", equatorial_coord.dec),
            l=("t", galactic_coord.l),
            b=("t", galactic_coord.b),
        )
        return self.data


def get_spectral_data(
    data_path: PathLike,
    topic_name: str,
    kisa_path: Optional[PathLike] = None,
    trans_coord: Optional[bool] = False,
    args: Optional[dict] = None,
    **kwargs: Optional[Any]
) -> xr.DataArray:
    """
    Notes
    -----
    Wall time:
    - 4s for trans_coord=False, 2deg x 2deg map
    - 78s for trans_coord=True, 2deg x 2deg map
    """
    ia = InitialArray(data_path, topic_name, kisa_path)
    ia.create_data_array()
    ret = ia.combine_metadata()  # kisa correction implicitly done here
    if not trans_coord:
        return ret
    ret = ia.convert_coordinates()
    if (not args) and (not kwargs):
        return ret
    return ia.channel2velocity(args, **kwargs)


def get_totalpower_data(
    data_path: PathLike,
    topic_name: str,
    kisa_path: Optional[PathLike] = None,
    trans_coord: Optional[bool] = False,
) -> xr.DataArray:
    """
    Notes
    -----
    Wall time:
    - 1s for trans_coord=False, 2deg x 2deg map
    - 73s for trans_coord=True, 2deg x 2deg map
    """
    ia = InitialArray(data_path, topic_name, kisa_path)
    ia.create_tp_array()
    ret = ia.combine_metadata()  # kisa correction implicitly done here
    if not trans_coord:
        return ret
    return ia.convert_coordinates()


class Initial_array(InitialArray):
    """Aliases for compatibility."""

    def __init__(self, data_path, kisa_path, topic_name):
        super().__init__(data_path, topic_name, kisa_path)
        # variable mappings
        self.topic = self.topic_name
        self.path = self.data_path
        self.path_to_kisa_param = self.kisa_path

    def get_data_array(self):
        self.create_data_array()
        # variable mappings
        self.data_array = self.data_set.spec
        self.obsmode_array = self.obsmode_set.obs_mode
        self.az_array = self.encoder_set.enc_az
        self.el_array = self.encoder_set.enc_el

        return (self.data_array, self.obsmode_array, self.az_array, self.el_array)

    def get_tp_array(self):
        self.create_tp_array()
        # variable mappings
        self.data_array = self.data_set.total_power
        self.obsmode_array = self.obsmode_set.obs_mode
        self.az_array = self.encoder_set.enc_az
        self.el_array = self.encoder_set.enc_el

        return (self.data_array, self.obsmode_array, self.az_array, self.el_array)

    def apply_kisa(self):
        self.correct_kisa()
        # variable mappings
        self.kisa_applyed_az = self.encoder_set.az
        self.kisa_applyed_el = self.encoder_set.el

        return (self.kisa_applyed_az, self.kisa_applyed_el)

    def concatenate(self):
        # variable mappings
        self.concatenated_array = self.combine_metadata(int_obsmode=True)
        self.obsmode_array_int = self.obsmode_set.int_obsmode

        return self.concatenated_array

    def ch2velo(self, arg_dict):
        self.channel2velocity(arg_dict)
        # variable mapping
        self.velo_list = self.data.v_lsr
        self.concatenated_array = self.data

        return self.velo_list

    def get_lb(self):
        self.convert_coordinates()
        # variable mappings
        self.l_list = self.data.l
        self.b_list = self.data.b
        self.ra_list = self.data.ra
        self.dec_list = self.data.dec

        return (self.l_list, self.b_list, self.ra_list, self.dec_list)

    def make_data_array(self):
        return get_spectral_data(self.path, self.topic_name, self.kisa_path)
