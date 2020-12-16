#!/usr/bin/env python3

from functools import wraps

from astropy.coordinates import EarthLocation
from astropy.coordinates import SkyCoord, CartesianDifferential, LSR
from astropy.time import Time
import astropy.constants as constants
import astropy.units as u
import numpy as np


class Doppler(object):
    """

    Calculate various quantities related to Doppler tracking.

    Parameters
    ----------
    spectrometer : str
        Spectrometer name, case insensitive. Either ``XFFTS`` or ``AC240`` are supported.
    rest_freq : astropy.units.quantity.Quantity
        Rest frequency of the line spectrum to be evaluated.
    species : str
        Name of the observed species; only CO isotopes (12CO_10, 13CO_10, C18O_10, 12CO_21, 13CO_21, C18O_21) are supported.
    LO1st_freq : astropy.units.quantity.Quantity
        Frequency of 1st local oscilator.
    LO1st_factor : int or float
        Factor of frequency multiplier for 1st local oscilator.
    LO2nd_freq : astropy.units.quantity.Quantity
        Frequency of 2nd local oscilator.
    obstime : float or astropy.units.quantity.Quantity
        Time the observation is done. If float value is given, it'll be interpreted as UNIX-time.
    ra : astropy.units.quantity.Quantity
        Right ascension of the target body.
    dec : astropy.units.quantity.Quantity
        Declination of the target body.

    Notes
    -----
    Parameters should be given as kwargs or dict. See examples below.

    Raises
    ------
    NameError
        When invalid argument name is passed.
    TypeError
        When non-named argument is passed.

    Examples
    --------
    >>> dp = Doppler(spectrometer='xffts', rest_freq=230.538*u.GHz, LO1st_freq=18.8*u.GHz, LO1st_factor=12, LO2nd_freq=4*u.GHz)
    >>> dp.heterodyne()
    <Quantity 0.938 GHz>
    >>> args = {'obstime': Time("2020-05-02T09:58:16.962", format="fits"),\
                'ra': 146.989193 * u.deg,\
                'dec': 13.278768 * u.deg}
    >>> dp.set_args(args)
    >>> dp.v_obs()
    <Quantity -36.03406182 km / s>

    """
    __VARS = ['spectrometer', 'rest_freq', 'LO1st_freq', 'LO1st_factor', 'LO2nd_freq', 'obstime', 'ra', 'dec', 'species']
    # observatory location
    LOC_NANTEN2 = EarthLocation(
        lon=-67.70308139 * u.deg,
        lat=-22.96995611 * u.deg,
        height=4863.85 * u.m,
    )
    SPECIES_to_REST_FREQ = {
        '12co_10': 115.271204 * u.GHz,
        '13co_10': 110.201353 * u.GHz,
        'c18o_10': 109.782173 * u.GHz,
        '12co_21': 230.538 * u.GHz,
        '13co_21': 220.398681 * u.GHz,
        'c18o_21': 219.560363 * u.GHz,
    }

    def __init__(self, args=None, **kwargs):
        # set instance variables
        if args:
            if isinstance(args, dict):
                kwargs.update(args)
            else:
                raise(TypeError("Arguments must be named. Use kwargs or dict."))
        self.set_args(**kwargs)

    def set_args(self, args=None, **kwargs):
        """Pass argument(s) to the instance.

        Examples
        --------
        >>> dp = Doppler()
        >>> dp.set_args(spectrometer='XFFTS')
        >>> dp.spectrometer
        'XFFTS'
        """
        if args:
            kwargs.update(args)
        for key, val in kwargs.items():
            if key in self.__VARS:
                setattr(self, key, val)
            else:
                raise(NameError(f'invalid argument : {key}'))
        return
    
    def species2freq(func):
        @wraps(func)
        def translate(inst, *args):
            if hasattr(inst, 'species'):
                inst.rest_freq = inst.SPECIES_to_REST_FREQ[inst.species.lower()]
            return func(inst, *args)
        return translate

    def __check_args(self, args):
        """

        Check if the parameters needed to run function are declared.

        Parameters
        ----------
        args : list[str]

        Raises
        ------
        AttributeError
            When undeclared parameter exists.

        """
        undeclared = []
        for arg in args:
            try:
                getattr(self, arg)
            except AttributeError:
                undeclared.append(arg)
        if undeclared:
            raise AttributeError(f'Parameter {undeclared} is not given. Use `set_args`')
        return

    @species2freq
    def heterodyne(self):
        """Where the line emission will appear.

        Spectrometer frequency where the line emission will appear. Since V_obs is not took into account, error of up to ~0.05 GHz is expected. For more precision, use `ch_speed`.

        Returns
        -------
        float
            Frequency on spectrometer the line emission will appear.

        Raises
        ------
        ValueError
            When invalid spectrometer name is given.

        Notes
        -----
        Parameters ['spectrometer', 'rest_freq', 'LO1st_freq', 'LO1st_factor', 'LO2nd_freq'] should be given.

        Examples
        --------
        >>> dp = Doppler(spectrometer='xffts', rest_freq=115.27*u.GHz, LO1st_freq=17.5*u.GHz, LO1st_factor=6, LO2nd_freq=9.5*u.GHz)
        >>> dp.heterodyne()
        <Quantity 0.77 GHz>

        """
        # check if necessary parameters have already been declared
        __vars = ['spectrometer', 'rest_freq', 'LO1st_freq', 'LO1st_factor', 'LO2nd_freq']
        self.__check_args(__vars)
        # set spectrometer constants
        if self.spectrometer.lower() == 'xffts':
            self.ch_num = 32768
            self.band_width = 2 * u.GHz
        elif self.spectrometer.lower() == 'ac240':
            self.ch_num = 16384
            self.band_width = 1 * u.GHz
        else:
            raise(ValueError(f"Invalid spectrometer name : {self.spectrometer}"))
        # calculate frequency shift
        freq_after1st = self.rest_freq - self.LO1st_freq * self.LO1st_factor
        self.sideband_factor = np.sign(freq_after1st)
        freq_after2nd = abs(freq_after1st) - self.LO2nd_freq
        self.sideband_factor *= np.sign(freq_after2nd)
        spec_freq = abs(freq_after2nd)
        return spec_freq

    def ch_speed(self):
        """Speed for each channel.

        Calculate recessional velocity relative to V_LSR for each channel of the spectrometer.

        Returns
        -------
        list of astropy.units.quantity.Quantity
            V_LSR for each channel.

        Notes
        -----
        Parameters ['spectrometer', 'rest_freq', 'LO1st_freq', 'LO1st_factor', 'LO2nd_freq', 'obstime', 'ra', 'dec'] should be given.

        Examples
        --------
        >>> args = {'spectrometer': 'xffts',\
                    'rest_freq': 115.271204*u.GHz,\
                    'LO1st_freq': 17.5*u.GHz,\
                    'LO1st_factor': 6,\
                    'LO2nd_freq': 9.5*u.GHz,\
                    'obstime': 1588413496.962337,\
                    'ra': 146.989193*u.deg,\
                    'dec': 13.278768 * u.deg}
        >>> dp = Doppler(args)
        >>> dp.ch_speed()
        <Quantity [ 1969.68059067,  1969.52185303,  1969.36311538, ...,
                   -3231.35836405, -3231.51710169, -3231.67583934] km / s>
        """
        spec_freq = self.heterodyne()
        speed_resolution = -1 * self.band_width / self.rest_freq / self.ch_num * constants.c
        v_0GHz = -1 * speed_resolution * spec_freq / (self.band_width / self.ch_num)
        v_base = np.array(range(self.ch_num)) * speed_resolution * self.sideband_factor
        v_apparent = v_base + v_0GHz
        v_obs = self.v_obs()
        v_lsr = v_apparent + v_obs
        return v_lsr.to(u.km/u.s)

    def v_obs(self):
        """Observer's verocity relative to LSR.

        Calcurate line-of-sight component of observer's velocity relative to
        LSR due to solar motion, the Earth's rotation and revolution.

        Returns
        -------
        astropy.units.quantity.Quantity
            Velocity relative to LSR, along the line of sight.

        Notes
        -----
        Parameters ['obstime', 'ra', 'dec'] should be given.
        All parameters can be given as lists.

        Examples
        --------
        >>> obs_time = Time("2020-05-02T09:58:16.962", format="fits")
        >>> Doppler(obstime=obs_time, ra=146.989193 * u.deg, dec=13.278768 * u.deg).v_obs()
        <Quantity -36.03406182 km / s>
        >>> dp = Doppler(obstime=[1588413496.962337, 1588413500.962337],\
                         ra=[146.989193 * u.deg, 147.989193 * u.deg],\
                         dec=[13.278768 * u.deg, 15.278768 * u.deg])
        >>> dp.v_obs()
        <Quantity [-36.03406101, -35.32385628] km / s>

        """
        # check if necessary parameters have already been declared
        __vars = ['obstime', 'ra', 'dec']
        self.__check_args(__vars)

        # galactic rotation parameter (solar motion)
        v_sun = 20 * u.km / u.s
        dir_sun = SkyCoord(
            ra=18 * 15 * u.deg,
            dec=30 * u.deg,
            frame="fk4",
            equinox=Time('B1900'),
        ).galactic
        U = v_sun * np.cos(dir_sun.b) * np.cos(dir_sun.l)
        V = v_sun * np.cos(dir_sun.b) * np.sin(dir_sun.l)
        W = v_sun * np.sin(dir_sun.b)
        v_bary = CartesianDifferential(U, V, W)

        # set target position
        if not isinstance(self.obstime, Time):
            self.obstime = Time(self.obstime, format='unix')
        target = SkyCoord(
            ra=self.ra,
            dec=self.dec,
            frame="fk5",
            obstime=self.obstime,
            location=self.loc_nanten2,
        )

        # calculate V_obs
        v_obs = SkyCoord(
            self.LOC_NANTEN2.get_gcrs(self.obstime)
        ).transform_to(
            LSR(v_bary=v_bary)
        ).velocity

        # normal component of V_obs to stellar body
        v_correction = v_obs.d_x * np.cos(target.icrs.dec) * np.cos(target.icrs.ra) + \
            v_obs.d_y * np.cos(target.icrs.dec) * np.sin(target.icrs.ra) + \
            v_obs.d_z * np.sin(target.icrs.dec)

        return v_correction


if __name__ == '__main__':
    # import doctest
    # doctest.testmod(verbose=True)
    pass
