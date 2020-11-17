from astropy.coordinates import EarthLocation, SkyCoord, CartesianDifferential, LSR
from astropy.time import Time
import astropy.units as u
import numpy as np


def v_obs(obstime, target_ra, target_dec):
    """Observer's verocity relative to LSR.

    Calcurate line-of-sight component of observer's velocity relative to LSR due to solar motion, the Earth's rotation and revolution.

    Parameters
    ----------
    obstime : float or astropy.time.core.Time
        Time the observation is done. If float value is given, it'll be interpreted as UNIX-time.
    target_ra : astropy.units.quantity.Quantity
        RightAscension of the target.
    target_dec : astropy.units.quantity.Quantity
        Declination of the target.

    Returns
    -------
    astropy.units.quantity.Quantity
        Velocity relative to LSR, along the line of sight.
    
    Notes
    -----
    All parameters can be given as lists.

    Examples
    --------
    >>> obs_time = Time("2020-05-02T09:58:16.962", format="fits")
    >>> v_obs(obs_time, 146.989193 * u.deg, 13.278768 * u.deg)
    <Quantity -36.03406182 km / s>
    >>> v_obs([1588413496.962337, 1588413500.962337], [146.989193 * u.deg, 147.989193 * u.deg], [13.278768 * u.deg, 15.278768 * u.deg])
    <Quantity [-36.03406101, -35.32385628] km / s>

    """
    
    # observatory location
    loc_nanten2 = EarthLocation(
        lon = -67.70308139 * u.deg,
        lat = -22.96995611 * u.deg,
        height = 4863.85 * u.m,
    )

    # galactic rotation parameter (solar motion)
    v_sun = 20 * u.km / u.s
    dir_sun = SkyCoord(
        ra = 18 * 15 * u.deg,
        dec = 30 * u.deg,
        frame = "fk4",
        equinox = Time('B1900'),
    ).galactic
    U = v_sun * np.cos(dir_sun.b) * np.cos(dir_sun.l)
    V = v_sun * np.cos(dir_sun.b) * np.sin(dir_sun.l)
    W = v_sun * np.sin(dir_sun.b)
    v_bary = CartesianDifferential(U, V, W)

    # set target position
    if isinstance(obstime, Time):
        t_obs = obstime
    else:
        t_obs = Time(obstime, format='unix')
    target = SkyCoord(
        ra = target_ra,
        dec =  target_dec,
        frame = "fk5",
        obstime = t_obs,
        location = loc_nanten2,
    )

    # calculate V_obs
    v_obs = SkyCoord(
        loc_nanten2.get_gcrs(t_obs)
    ).transform_to(
        LSR(v_bary=v_bary)
    ).velocity

    # normal component of V_obs to stellar body
    v_correction = v_obs.d_x * np.cos(target.icrs.dec) * np.cos(target.icrs.ra) + \
                   v_obs.d_y * np.cos(target.icrs.dec) * np.sin(target.icrs.ra) + \
                   v_obs.d_z * np.sin(target.icrs.dec)
    
    return v_correction


if __name__ == "__main__":
    pass
