import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from astropy import coordinates as co
from astropy.time import Time
import astropy.units as u
from nasco_analysis import io

def get_data(path, kisa_param , xFFTS_Data_topics):
    obj = io.Initial_array(
        path_to_data = path,
        path_to_kisa_param = kisa_param,
        xFFTS_Data_topic = xFFTS_Data_topics
    )
    obj.get_data_array()
    obj.apply_kisa()
    raw_array = obj.concatenate()

    return raw_array

def separate(array):
    ON_OL = array['obsmode'] == 2.0
    ON = array[ON_OL]
    OFF_OL = array['obsmode'] == 1.0
    OFF = array[OFF_OL]
    HOT_OL = array['obsmode'] == 0.0
    HOT = array[HOT_OL]

    return ON, OFF, HOT

def mean(array):
    mean = array.mean("t", keepdims=True)
    mean["t"] = array.t.mean("t", keepdims=True)
    return mean

def scanmask(array):
    scan_num_list = np.unique(array['scan_num'])
    
    mean_array_list = []
    
    for scan_num in scan_num_list:
        scanmask = array['scan_num'] == scan_num
        scanmasked_array = array[scanmask]        
        scanmasked_array_mean = mean(scanmasked_array)
        
        mean_array_list.append(scanmasked_array_mean)
    mean_array = xr.concat(mean_array_list, dim='t')
    return mean_array

def chopper_wheel(on_array, off_array, hot_array):
    fixed_off_array = scanmask(off_array)
    fixed_hot_array = scanmask(hot_array)
    reindexed_off_array = fixed_off_array.interp_like(on_array)
    reindexed_hot_array = fixed_hot_array.interp_like(on_array)
    
    calib_array = (on_array-reindexed_off_array) / (reindexed_hot_array-reindexed_off_array)*300
    
    return calib_array

def make_total_power_array(array, law_ch_num='none', high_ch_num='none'):
    if law_ch_num == 'none':
        ON_TP = np.sum(array, axis=1)
        total_power_array = ON_TP * 0.15
    else:    
        ON_TP = np.sum(array[:,law_ch_num:high_ch_num], axis=1)
        len_ch_num = len(array[2])
        total_power_array = ON_TP * len_ch_num/(high_ch_num-law_ch_num) * 0.15
        
    return total_power_array

def make_pix_array(array, Num_pix):
    
    nanten2 = co.EarthLocation(
        lon = -67.70308139 * u.deg,
        lat = -22.96995611  * u.deg,
        height = 4863.85 * u.m)
    
    time_unix_list = np.array(array['t'])
    time_unix_list_astype = Time(time_unix_list, format='datetime64')
    jupiter_radec_list = co.get_body('Jupiter', time_unix_list_astype)
    jupiter_radec_list.location = nanten2
    jupiter_azel_list = jupiter_radec_list.transform_to(co.AltAz())
    paz = jupiter_azel_list.az.deg - 360
    pel = jupiter_azel_list.alt.deg
    
    delta_az = paz - array['azlist']
    delta_el = pel - array['ellist']
    
    daz_min = delta_az.min()
    del_min = delta_el.min()
    
    daz_mask_ = delta_az - daz_min
    del_mask_ = delta_el - del_min
    
    daz_max = daz_mask_.max()
    del_max = del_mask_.max()
    
    daz_mask = daz_mask_ * (Num_pix-1) / daz_max
    del_mask = del_mask_ * (Num_pix-1) / del_max
    
    pix_array = xr.DataArray(
        np.array(array),
        dims=['t'],
        coords={'t':array['t'],
                
                'daz':('t',np.array(delta_az)),
                'del':('t',np.array(delta_el)),
                'daz_pix':('t',np.array(daz_mask)),
                'del_pix':('t',np.array(del_mask))
            
                }
    )
    
    return pix_array

def make_grid(Num_pix):
    x = np.arange(Num_pix)
    y = np.arange(Num_pix)
    X, Y = np.meshgrid(x,y)
    return X, Y

def convolution(array, Num_pix):
    X, Y = make_grid(Num_pix)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    XY = np.c_[X_flat, Y_flat]
    conved_array_ = []
    for XYv in XY:
        r = (XYv[0] - array['daz_pix'])**2 + (XYv[1] - array['del_pix'])**2
        wei_dis = np.exp(-r)
        value = np.average(array, weights=wei_dis)
        conved_array_.append(value)
    conved_array_ = np.array(conved_array_)
    conved_array = conved_array_.reshape([Num_pix, Num_pix])
    return conved_array

def make_figure(array, Num_pix, law_ch_num='none', high_ch_num='none', save='False'):

    X, Y = make_grid(Num_pix)

    fig = plt.figure(figsize=[14,5])
    if law_ch_num == 'none':
        fig.suptitle('Beam pattern', fontsize=18)
    else:
        fig.suptitle('Beam pattern_ch_num:' + str(law_ch_num) + 'to' + str(high_ch_num), fontsize=18)

    ax1 = fig.add_subplot(121)
    cont = ax1.contour(X,Y,array, colors=['white'], linewidths=0.7)
    map1 = ax1.pcolormesh(X, Y, array, cmap='inferno')
    ax1.set_xlabel('$\mathregular{\Delta Az_{pix}}$',fontsize=16)
    ax1.set_ylabel('$\mathregular{\Delta El_{pix}}$',fontsize=16)
    ax1.set_title('with contour',fontsize=18)
    cbar1 = fig.colorbar(map1)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label('K km/s', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax2 = fig.add_subplot(122)
    map2 = ax2.pcolormesh(X, Y, array, cmap='inferno')
    ax2.set_xlabel('$\mathregular{\Delta Az_{pix}}$',fontsize=16)
    ax2.set_ylabel('$\mathregular{\Delta El_{pix}}$',fontsize=16)
    ax2.set_title('non contour',fontsize=18)
    cbar2 = fig.colorbar(map2)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label('K km/s', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    fig.subplots_adjust(wspace=0.3)
    
    if save == True:
        if law_ch_num == 'none':
            fig.savefig('Beam pattern.png')
        else:
            fig.savefig('Beam pattern_ch_num:' + str(law_ch_num) + 'to' + str(high_ch_num) + '.png') 
        print('Save completed')
    else:
        print('Did not save')

def Beam_pattern(path, kisa_param , xFFTS_Data_topics, Num_pix, law_ch_num='none', high_ch_num='none', save='False'):
    raw_array = get_data(path, kisa_param , xFFTS_Data_topics)
    ON, OFF, HOT = separate(raw_array)
    calib_array = chopper_wheel(ON, OFF, HOT)
    total_power_array = make_total_power_array(calib_array, law_ch_num, high_ch_num)
    pix_array = make_pix_array(total_power_array, Num_pix)
    conved_array = convolution(pix_array, Num_pix)
    make_figure(conved_array, Num_pix, law_ch_num, high_ch_num, save)