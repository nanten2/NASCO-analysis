from astropy.coordinates import SkyCoord
import numpy as np
import astropy.coordinates
from astropy.coordinates import AltAz, FK5, Galactic
import astropy.units as u
import astropy.time
import astropy.constants as c
import xarray as xr
import necstdb
from datetime import datetime
from tqdm import tqdm
from .kisa_rev import apply_kisa_test
from joblib import Parallel, delayed

def get_array(path, xFFTS_Data_topic):
    db = necstdb.opendb(path)
    xFFTS_data = db.open_table(xFFTS_Data_topic).read(astype='array')
    obsmode =  db.open_table('obsmode').read(astype='array')
    enc =  db.open_table('status_encoder').read(astype='array')
    
    spec_array = xr.DataArray(

        xFFTS_data['spec'], 
        dims=['t', 'spectral_data'], 
        coords={'t':xFFTS_data['timestamp']}
    )


    obsmode_array = xr.DataArray(

        obsmode['obs_mode'],
        dims = ['t'],
        coords={'t':obsmode['received_time'], 'scan_num':('t', obsmode['scan_num'])}


    )
    
    az_array = xr.DataArray(
        
        enc['enc_az']/3600, 
        dims=['t'],
        coords={'t':enc['timestamp']}
    )
    
    el_array = xr.DataArray(
        
        enc['enc_el']/3600, 
        dims=['t'],
        coords={'t':enc['timestamp']}
    )

    return spec_array, obsmode_array, az_array, el_array

def apply_kisa(az_array, el_array, path_to_hosei_file):
    delta_az_el = [apply_kisa_test(np.deg2rad(az), np.deg2rad(el), path_to_hosei_file) for az, el in zip(az_array.values, el_array.values)]    
    kisa_param = np.array(delta_az_el).T
    delta_az = kisa_param[0]/3600
    delta_el = kisa_param[1]/3600

    kisa_applyed_az = az_array + delta_az
    kisa_applyed_el = el_array + delta_el
    
    return kisa_applyed_az, kisa_applyed_el

def concatenate(spec_array, obsmode_array, az_array, el_array):
    
    obsmode_number_array = []
    for obsmode in obsmode_array:
        mode_number = np.nan
        if obsmode == b'HOT       ':
            mode_number = 0.
        elif obsmode == b'OFF       ':
            mode_number = 1.
        elif obsmode == b'ON        ':
            mode_number = 2.
        else:
            pass
        obsmode_number_array.append(mode_number)
    
    obsmode_array_int = xr.DataArray(np.array(obsmode_number_array),dims=['t'],coords={'t':obsmode_array['t']})
    
    reindexed_scannum_array = obsmode_array.reindex(t=spec_array['t'], method='pad')
    reindexed_obsmode_array = obsmode_array_int.interp_like(spec_array)
    reindexed_encoder_az_array = az_array.interp_like(spec_array)
    reindexed_encoder_el_array = el_array.interp_like(spec_array)
    
    del obsmode_array
    del az_array
    del el_array
    raw_array = xr.DataArray(
        np.array(spec_array),
        dims=['t', 'spectral_data'],
        coords={'t':spec_array['t'],
                
               'obsmode':('t',np.array(reindexed_obsmode_array)),
               'scan_num':('t', np.array(reindexed_scannum_array['scan_num'])),
               'azlist':('t', np.array(reindexed_encoder_az_array)),
               'ellist':('t', np.array(reindexed_encoder_el_array))
                
               }
    )
    
    return raw_array

def get_lb(raw_array):
    
    time = [datetime.utcfromtimestamp(t) for t in np.array(raw_array['t'])]
    
    nanten2 = astropy.coordinates.EarthLocation(
        
            lon =  -67.70308139 * u.deg,
            lat = -22.96995611  * u.deg,
            height = 4863.85 * u.m    
    )
    
    location = nanten2
    
    AltAzcoordiantes = astropy.coordinates.SkyCoord(
        
        az=raw_array['azlist'], 
        alt=raw_array['ellist'], 
        frame='altaz', 
        obstime=time,
        location=location, 
        unit='deg')
    
    l_list = AltAzcoordiantes.transform_to(Galactic).l
    b_list = AltAzcoordiantes.transform_to(Galactic).b
    ra_list = AltAzcoordiantes.transform_to('fk5').ra
    dec_list = AltAzcoordiantes.transform_to('fk5').dec
    return l_list, b_list, ra_list, dec_list

def make_data_array(raw_array, l_list, b_list, ra_list, dec_list, IF):
    data_array = xr.DataArray(
        np.array(raw_array), 
        dims=['t', 'spectral_data'],
        coords={'t':raw_array['t'],
               'obsmode':('t',np.array(raw_array['obsmode'])),
               'scan_num':('t', np.array(raw_array['scan_num'])),
               'l_list':('t', l_list),
               'b_list':('t', b_list),
               'ra_list':('t', ra_list),
               'dec_list':('t',dec_list)}
    )

    return data_array