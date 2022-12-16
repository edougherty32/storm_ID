#!/usr/bin/env python
# coding: utf-8

# ### script for classifiying stratiform rainfall
# ### of 3 hourly reflectivity output from GridRad
# ### date created: 15 September 2021
# ### author: Erin Dougherty (doughert@ucar.edu)

# In[3]:


import math 
import numpy as np
import pandas as pd
import matplotlib as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from netCDF4 import Dataset, num2date
from datetime import datetime, date, timedelta
import glob
import xarray as xr
import geopandas as gp
#from wrf import getvar, ALL_TIMES


# ### open test reflectivity output

# In[70]:


print('opening data')


# In[5]:


radar_path = '/glade/scratch/doughert/grid_refl_out/'
test_dbz = xr.open_dataset(radar_path + 'gridrad_refl_1995_1996_3h.nc' )


# ### MCS identification

# #### mask all values below 40 dbz

# In[7]:


mask_dbz = test_dbz['refl0'].where((test_dbz['refl0']< 40.0) & (test_dbz['refl0']> 18.0))

object_id_01s = mask_dbz.notnull().astype(int)


# #### ID objects based on contiguous reflectivity > 40dBZ

# In[8]:


from scipy.ndimage import label

object_ar = []
num_feats = []

for i in range(len(object_id_01s)):
    labeled_array, num_features = label(object_id_01s[i])
    object_ar.append(labeled_array)
    num_feats.append(num_features)


# #### count number of unique features in an array

# In[10]:


unique_obs = []
counts = []

for t in range(len(test_dbz['refl0'])):
    uniqueo, cts, = np.unique(object_ar[t], return_counts=True, )
    unique_obs.append(uniqueo)
    counts.append(cts)

count_unique_objs = []
for i in range(len(counts)):
    count_unique_objs.append(np.asarray((unique_obs[i], counts[i])).T)
    
## delete first row in array (which corresponds to # of objects w/ 0 pcp)
for i in range(len(count_unique_objs)):
    count_unique_objs[i] = np.delete(count_unique_objs[i],0,0)
    counts[i] = np.delete(counts[i],0)
    unique_obs[i] = np.delete(unique_obs[i],0)


# #### find where objects are over 40 dBZ and > 100 km 

# In[12]:


obj_100km_idx = []

# find index of objects w/ counts(grid cells) > 100 km (by MCS definition)
for i in range(len(count_unique_objs)):
    idx_100km = np.where(counts[i]>50)
    obj_100km_idx.append(idx_100km)
    
# get unique objs and # of cells corresponding to criteria
obj_id_100km = []
obj_numcells_100km = []

for i in range(len(count_unique_objs)):
    obj_id = unique_obs[i][np.where(counts[i]>50)]
    num_cells = counts[i][np.where(counts[i]>50)]
    obj_id_100km.append(obj_id)
    obj_numcells_100km.append(num_cells)


# ### mask refl values based on where object array == object array indices where >40 dbZ and >100 km

# In[14]:


mask_40dbz_100 = [np.isin(object_ar[i], obj_id_100km[i]).astype(int) for i in range(len(object_ar))]
mask_refl_mcs = [test_dbz['refl0'][i].where(mask_40dbz_100[i]==1) for i in range(len(object_ar))]


# ### get size of MCSs

# #### get x and y indices of object id values (i.e., where >40 dbz and 100 km)

# In[1]:


print('getting size of isolated convective cells')


# In[17]:


mcs_idxs_xs = []
mcs_idxs_ys = []

for i in range(len(object_ar)):
    x_locs = np.nonzero(np.isin(object_ar[i],obj_id_100km[i]))[0]
    y_locs = np.nonzero(np.isin(object_ar[i],obj_id_100km[i]))[1]
    mcs_idxs_xs.append(x_locs)
    mcs_idxs_ys.append(y_locs)


# #### get number of grid cells corresponding to the index of each unique object

# In[19]:


import itertools

fill = []
fill_idxs = []

for i in range(len(object_ar)):
    fill1 = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(obj_numcells_100km[i], obj_numcells_100km[i]))))
    fill1_idxs = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(obj_id_100km[i], obj_numcells_100km[i]))))
    fill.append(fill1)
    fill_idxs.append(fill1_idxs)


# In[21]:


mcs_idx_xs_flat = [list(mcs_idxs_xs[i]) for i in range(len(object_ar))]
mcs_idx_ys_flat = [list(mcs_idxs_ys[i]) for i in range(len(object_ar))]


# #### create empty array in same shape of reflectivity arrray
# #### and fill with grid size of objects based on idex (do the same with different object ids)

# In[23]:


mcs_size = np.zeros((len(test_dbz['refl0']), 1201,2301))
mcs_id = np.zeros((len(test_dbz['refl0']), 1201,2301))


# In[24]:


for i in range(len(object_ar)):
    mcs_size[i, mcs_idx_xs_flat[i], mcs_idx_ys_flat[i]]=fill[i]
    mcs_id[i, mcs_idx_xs_flat[i], mcs_idx_ys_flat[i]]=fill_idxs[i]


# ### compute average and max reflectivity for each MCS

# #### only get values at indices for MCS objects

# In[27]:


refl_obj_vals = []
mcs_id_obj = []
mcs_size_obj = []

for i in range(len(object_ar)):
    refl_ob = mask_refl_mcs[i].values[mcs_idx_xs_flat[i], mcs_idx_ys_flat[i]]
    id_ob = mcs_id[i, mcs_idx_xs_flat[i], mcs_idx_ys_flat[i]]
    size_ob = mcs_size[i, mcs_idx_xs_flat[i], mcs_idx_ys_flat[i]]
    refl_obj_vals.append(refl_ob)
    mcs_id_obj.append(id_ob)
    mcs_size_obj.append(size_ob)


# #### get indices of where unique values start and stop

# In[29]:


unique_idxs =[]
idx = []

for i in range(len(object_ar)):
    unique, index = np.unique(mcs_id_obj[i], return_index=True)
    unique_idxs.append(unique)
    idx.append(index)


# #### compute max and avg refl values for each object

# In[31]:


avg_mcs_dbz = []
max_mcs_dbz = []

for l in range(len(object_ar)):
    mean_refl = np.zeros(len(idx[l]))
    max_refl = np.zeros(len(idx[l]))
    for i in range(len(idx[l])):
        if ((i < len(idx[l])-1) and (len(idx[l])>1)):
            idxn = idx[l][i]
            idx2 = idx[l][i+1]
            mean_refl[i] = np.nanmean(refl_obj_vals[l][idxn:idx2])
            max_refl[i] = np.nanmax(refl_obj_vals[l][idxn:idx2]) 
        elif ((i == len(idx[l])-1) or (len(idx[l])==1)):
            idxn = idx[l][i]
            mean_refl[i] = np.nanmean(refl_obj_vals[l][idxn])
            max_refl[i] = np.nanmax(refl_obj_vals[l][idxn])
    avg_mcs_dbz.append(mean_refl)
    max_mcs_dbz.append(max_refl)


# ### make into df 

# In[33]:


unique_idxs_int = [unique_idxs[i].astype(int) for i in range(len(avg_mcs_dbz))]


# In[37]:


times_mcs_sh = [mask_refl_mcs[i].time.values for i in range(len(mask_refl_mcs))]
times_mcs_repeat = [np.repeat(times_mcs_sh[i], len(unique_idxs_int[i])) for i in range(len(unique_idxs_int))]


# In[38]:


mcs_df = pd.DataFrame({'object_idx':unique_idxs_int, 'size (km)':obj_numcells_100km, 'avg_dbz':avg_mcs_dbz,
                       'max_dbz':max_mcs_dbz, 'time':times_mcs_repeat})


# #### expand rows of arrays to larger df 

# In[39]:


object_idx_long = [x for sublist in mcs_df['object_idx'] for x in sublist]
size_mcs = [x for sublist in mcs_df['size (km)'] for x in sublist]
avg_dbz_mcs = [x for sublist in mcs_df['avg_dbz'] for x in sublist]
max_dbz_mcs = [x for sublist in mcs_df['max_dbz'] for x in sublist]
times_mcs = [x for sublist in mcs_df['time'] for x in sublist]


# In[2]:


print('number of all isolated convective objects')
print(len(object_idx_long))


# ### make new expanded df

# In[41]:


mcs_df2 = pd.DataFrame({'object_idx':object_idx_long, 'size (km)':size_mcs, 'avg_dbz':avg_dbz_mcs,
                        'max_dbz':max_dbz_mcs, 'time':times_mcs})


# ### find where isolate conv cores fall into Bukovsky regions

# #### import Bukovsky climate regions

# In[44]:


from cartopy.io import shapereader
import geopandas as gp
from geopandas import GeoDataFrame

conus_reg = gp.GeoDataFrame.from_file( "/glade/work/doughert/asp/flood_storm_types/Bukovsky_conus_no_overlap/Bukovsky_conus_no_overlap.shp")
conus_reg.head()
print(len(conus_reg))

regions = conus_reg['geometry'].values


# In[45]:


conus_reg['region_list'] = ['Pac_NW', 'Pac_SW', 'MountainW', 'Desert', 'GP', 'Praire', 'South', 'Lakes', 'East']


# #### turn iso conv cores into polygons

# In[46]:


import rasterio.features

mask_refl_mcs_0s = [mask_refl_mcs[i].fillna(0) for i in range(len(mask_refl_mcs))]
mask_refl_mcs_01s = [mask_refl_mcs_0s[i].where(mask_refl_mcs_0s[i]==0, other=1) for i in range(len(mask_refl_mcs))]
mask_refl_mcs_int = [mask_refl_mcs_01s[i].values.astype(rasterio.int32) for i in range(len(mask_refl_mcs))]


# In[47]:


import rasterio
import shapely

shapes_mcs = [rasterio.features.shapes(mask_refl_mcs_int[i]) for i in range(len(mask_refl_mcs))]


# In[48]:


polygons_mcs = []

for i in range(len(mask_refl_mcs)):
    poly = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes_mcs[i] if shape[1] == 1]
    polygons_mcs.append(poly)


# #### get polygon coords

# In[50]:


x_coords = []
y_coords = []
times_all_mcs = []

for i in range(len(mask_refl_mcs)):
    for p in range(len(polygons_mcs[i])):
        x,y = polygons_mcs[i][p].exterior.coords.xy
        x_coords.append(np.array(x).astype(int))
        y_coords.append(np.array(y).astype(int))
        times_all_mcs.append(times_mcs_sh[i])                                                  


# #### get lat/lon corresponding to polygon coords

# In[52]:


poly_lon = []

for c in range(len(x_coords)):
    lon_idx = x_coords[c]
    lon_idx[np.where(lon_idx==241)]=240
    lon_idx2 = np.where(lon_idx==2301, 2300, lon_idx)
    lon_vals = mask_refl_mcs[0].lon.values[lon_idx2]-360
    poly_lon.append(lon_vals)


# In[53]:


poly_lat = []

for c in range(len(y_coords)):
    lat_idx = y_coords[c]
    lat_idx2 = np.where(lat_idx==1201, 1200, lat_idx)
    lat_vals = mask_refl_mcs[0].lat.values[lat_idx2]
    poly_lat.append(lat_vals)


# In[54]:


from shapely.geometry import Polygon
polygon_geom_mcs = [Polygon(zip(poly_lon[c], poly_lat[c])) for c in range(len(y_coords))]


# #### turn into new polygons w/ lat/lon coords

# In[56]:


geo_idx_mcs = np.arange(0, len(times_all_mcs), 1)


# In[57]:


from geopandas import GeoSeries, GeoDataFrame

crs = {'init': 'epsg:4326'}
mcs_polygon = gp.GeoDataFrame(index=geo_idx_mcs, crs=crs, geometry=GeoSeries(polygon_geom_mcs))    
mcs_polygon['time'] = times_all_mcs


# ### designate region as having isolated convective rainfall if convective core intersects region

# In[3]:


print('find which regions MCS fall into-this takes several hours')


# In[59]:


mcs_region = []
mcs_date = []
mcs_count = []
mcs_id = []

for f in range(len(conus_reg)):
    for t in range(len(mcs_polygon)):
        if (conus_reg['geometry'].iloc[f].intersects(mcs_polygon['geometry'].iloc[t].buffer(0.02))):
            mcs_region.append(conus_reg['region_list'].iloc[f])
            mcs_date.append(mcs_polygon['time'].iloc[t])
            mcs_count.append(1)
            mcs_id.append(mcs_df2['object_idx'].iloc[t])


# ### make a df of isolate convective core regions

# In[60]:


mcs_region = pd.DataFrame({'region':mcs_region, 'time':mcs_date, 'occurrence':mcs_count, 'object_idx':mcs_id})


# #### turn time objects into datetimes

# In[61]:


mcs_region['time'] = [pd.to_datetime(mcs_region['time'].iloc[i], format='%Y%m%d'+'T'+'%H:%M:%S') for i in range(len(mcs_region))]
mcs_df2['time'] = [pd.to_datetime(mcs_df2['time'].iloc[i], format='%Y%m%d'+'T'+'%H:%M:%S') for i in range(len(mcs_df2))]


# In[62]:


mcs_region['time_int'] =mcs_region['time'].astype(int)/10**9
mcs_df2['time_int'] = mcs_df2['time'].astype(int)/10**9


# In[63]:


mcs_region_sort = mcs_region.sort_values(by='time')
mcs_df2_sort = mcs_df2.sort_values(by='time')


# ### merge this df with df of convective characteristics

# In[64]:


mcs_all = mcs_region_sort.merge(mcs_df2_sort, how='left', on=['object_idx', 'time_int'] )


# In[65]:


mcs_all = mcs_all.drop(['time_int','time_y'],axis=1)


# ### resample to daily resolution

# In[67]:


mcs_daily = mcs_all.groupby('region').resample('D', on='time_x').mean().reset_index()


# In[69]:


print('length of MCSs')
print(len(mcs_daily))


# In[68]:


mcs_daily.to_csv('/glade/scratch/doughert/grid_refl_out/non_convect_daily_bukovsky_regions_1995_1996.csv')


# In[ ]:


print('export successful')

