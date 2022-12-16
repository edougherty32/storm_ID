#!/usr/bin/env python
# coding: utf-8

# ### script for classifiying convective type 
# ### of 3 hourly reflectivity output from GridRad
# ### date created: 15 September 2021
# ### author: Erin Dougherty (doughert@ucar.edu)

# In[1]:


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
import geopandas
#from wrf import getvar, ALL_TIMES


# ### open test reflectivity output

# #### path to radar output

# In[1]:


print('opening data')


# In[2]:


radar_path = '/glade/scratch/doughert/grid_refl_out/'


# In[3]:


radar_1995_96 = xr.open_dataset(radar_path + 'gridrad_refl_1995_1996_3h.nc' )


# ### MCS identification

# #### mask all values below 40 dbz

# In[5]:


mask_dbz = radar_1995_96['refl0'].where(radar_1995_96['refl0']> 40.0)

object_id_01s = mask_dbz.notnull().astype(int)


# #### ID objects based on contiguous reflectivity > 40dBZ

# In[6]:


from scipy.ndimage import label

object_ar = []
num_feats = []

for i in range(len(object_id_01s)):
    labeled_array, num_features = label(object_id_01s[i])
    object_ar.append(labeled_array)
    num_feats.append(num_features)


# #### count number of unique features in an array

# In[8]:


unique_obs = []
counts = []

for t in range(len(radar_1995_96['refl0'])):
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


# In[9]:


obj_100km_idx = []

# find index of objects w/ counts(grid cells) > 100 km (by MCS definition)
for i in range(len(count_unique_objs)):
    idx_100km = np.where(counts[i]>100)
    obj_100km_idx.append(idx_100km)
    
# get unique objs and # of cells corresponding to criteria
obj_id_100km = []
obj_numcells_100km = []

for i in range(len(count_unique_objs)):
    obj_id = unique_obs[i][np.where(counts[i]>100)]
    num_cells = counts[i][np.where(counts[i]>100)]
    obj_id_100km.append(obj_id)
    obj_numcells_100km.append(num_cells)


# #### find where objects are over 40 dBZ but < 100 km 

# In[10]:


obj_no100_idx = []

for i in range(len(count_unique_objs)):
    idx_small = np.where(counts[i]<100) and np.where(counts[i]>5)
    obj_no100_idx.append(idx_small)
    
# get unique objs and # of cells corresponding to criteria
obj_id_no100 = []
obj_numcells_no100 = []

for i in range(len(count_unique_objs)):
    obj_id = unique_obs[i][np.where(counts[i]<100) and np.where(counts[i]>5)]
    num_cells = counts[i][np.where(counts[i]<100) and np.where(counts[i]>5)]
    obj_id_no100.append(obj_id)
    obj_numcells_no100.append(num_cells)


# ### mask refl values based on where object array == object array indices where >40 dbZ but <100 km

# In[12]:


mask_40dbz_no100 = [np.isin(object_ar[i], obj_id_no100[i]).astype(int) for i in range(len(object_ar))]
mask_refl_iso_conv = [radar_1995_96['refl0'][i].where(mask_40dbz_no100[i]==1) for i in range(len(object_ar))]


# ### get size of isolated convective cells 

# #### get x and y indices of object id values (i.e., where >40 dbz and <100 km)

# In[2]:


print('getting size of isolated convective cells')


# In[14]:


iso_conv_idxs_xs = []
iso_conv_idxs_ys = []

for i in range(len(object_ar)):
    x_locs = np.nonzero(np.isin(object_ar[i],obj_id_no100[i]))[0]
    y_locs = np.nonzero(np.isin(object_ar[i],obj_id_no100[i]))[1]
    iso_conv_idxs_xs.append(x_locs)
    iso_conv_idxs_ys.append(y_locs)


# #### get number of grid cells corresponding to the index of each unique object

# In[16]:


import itertools

fill = []
fill_idxs = []

for i in range(len(object_ar)):
    fill1 = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(obj_numcells_no100[i], obj_numcells_no100[i]))))
    fill1_idxs = list(itertools.chain(*(itertools.repeat(elem, n) for elem, n in zip(obj_id_no100[i], obj_numcells_no100[i]))))
    fill.append(fill1)
    fill_idxs.append(fill1_idxs)


# In[17]:


iso_conv_idx_xs_flat = [list(iso_conv_idxs_xs[i]) for i in range(len(object_ar))]
iso_conv_idx_ys_flat = [list(iso_conv_idxs_ys[i]) for i in range(len(object_ar))]


# #### create empty array in same shape of reflectivity arrray
# #### and fill with grid size of objects based on idex (do the same with different object ids)

# In[18]:


iso_conv_size = np.zeros((len(radar_1995_96['refl0']), 1201,2301))
iso_conv_id = np.zeros((len(radar_1995_96['refl0']), 1201,2301))


# In[19]:


for i in range(len(object_ar)):
    iso_conv_size[i, iso_conv_idx_xs_flat[i], iso_conv_idx_ys_flat[i]]=fill[i]
    iso_conv_id[i, iso_conv_idx_xs_flat[i], iso_conv_idx_ys_flat[i]]=fill_idxs[i]


# ### compute average and max reflectivity for each isolated convective cell

# #### only get values at indices for isolated convective core objects

# In[21]:


refl_obj_vals = []
iso_conv_id_obj = []
iso_conv_size_obj = []

for i in range(len(object_ar)):
    refl_ob = mask_refl_iso_conv[i].values[iso_conv_idx_xs_flat[i], iso_conv_idx_ys_flat[i]]
    id_ob = iso_conv_id[i, iso_conv_idx_xs_flat[i], iso_conv_idx_ys_flat[i]]
    size_ob = iso_conv_size[i, iso_conv_idx_xs_flat[i], iso_conv_idx_ys_flat[i]]
    refl_obj_vals.append(refl_ob)
    iso_conv_id_obj.append(id_ob)
    iso_conv_size_obj.append(size_ob)


# #### get indices of where unique values start and stop

# In[23]:


unique_idxs =[]
idx = []

for i in range(len(object_ar)):
    unique, index = np.unique(iso_conv_id_obj[i], return_index=True)
    unique_idxs.append(unique)
    idx.append(index)


# #### compute max and avg refl values for each object

# In[24]:


avg_iso_core_dbz = []
max_iso_core_dbz = []

for l in range(len(object_ar)):
#for l in range(len(object_ar[0])):
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
    avg_iso_core_dbz.append(mean_refl)
    max_iso_core_dbz.append(max_refl)


# In[32]:


# avg_iso_core_dbz = []
# max_iso_core_dbz = []

# for i in range(len(idx)):
#     if i < 68:
#         mean_val = np.nanmean(refl_obj_vals[idx[i]:idx[i+1]])
#         max_val = np.nanmax(refl_obj_vals[idx[i]:idx[i+1]])
#         avg_iso_core_dbz.append(mean_val)
#         max_iso_core_dbz.append(max_val)   
#     else:
#          mean_val = np.nanmean(refl_obj_vals[idx[i]])
#          max_val = np.nanmax(refl_obj_vals[idx[i]])
#          avg_iso_core_dbz.append(mean_val)
#          max_iso_core_dbz.append(max_val)  


# ### make into df 

# In[27]:


unique_idxs_int = [unique_idxs[i].astype(int) for i in range(len(avg_iso_core_dbz))]


# In[28]:


times_iso_conv_core = [mask_refl_iso_conv[i].time.values for i in range(len(mask_refl_iso_conv))]
times_iso_core_repeat = [np.repeat(times_iso_conv_core[i], len(unique_idxs_int[i])) for i in range(len(unique_idxs_int))]


# In[29]:


iso_conv_core_df = pd.DataFrame({'object_idx':unique_idxs_int, 'size (km)':obj_numcells_no100, 'avg_dbz':avg_iso_core_dbz,
                                 'max_dbz':max_iso_core_dbz, 'time':times_iso_core_repeat})


# #### expand rows of arrays to larger df 

# In[30]:


object_idx_long = [x for sublist in iso_conv_core_df['object_idx'] for x in sublist]
size_iso_core = [x for sublist in iso_conv_core_df['size (km)'] for x in sublist]
avg_dbz_iso_core = [x for sublist in iso_conv_core_df['avg_dbz'] for x in sublist]
max_dbz_iso_core = [x for sublist in iso_conv_core_df['max_dbz'] for x in sublist]
times_iso_core = [x for sublist in iso_conv_core_df['time'] for x in sublist]


# In[31]:


print('number of all isolated convective objects')
print(len(object_idx_long))


# ### make new expanded df

# In[32]:


iso_conv_core_df2 = pd.DataFrame({'object_idx':object_idx_long, 'size (km)':size_iso_core, 'avg_dbz':avg_dbz_iso_core,
                                 'max_dbz':max_dbz_iso_core, 'time':times_iso_core})


# ### find where isolate conv cores fall into Bukovsky regions

# #### import Bukovsky climate regions

# In[34]:


from cartopy.io import shapereader
import geopandas as gp
from geopandas import GeoDataFrame

conus_reg = gp.GeoDataFrame.from_file( "/glade/work/doughert/asp/flood_storm_types/Bukovsky_conus_no_overlap/Bukovsky_conus_no_overlap.shp")
conus_reg.head()
print(len(conus_reg))

regions = conus_reg['geometry'].values


# In[35]:


conus_reg['region_list'] = ['Pac_NW', 'Pac_SW', 'MountainW', 'Desert', 'GP', 'Praire', 'South', 'Lakes', 'East']


# #### turn iso conv cores into polygons

# In[36]:


import rasterio.features

mask_refl_iso_conv_0s = [mask_refl_iso_conv[i].fillna(0) for i in range(len(mask_refl_iso_conv))]
mask_refl_iso_conv_01s = [mask_refl_iso_conv_0s[i].where(mask_refl_iso_conv_0s[i]==0, other=1) for i in range(len(mask_refl_iso_conv))]
mask_refl_iso_conv_int = [mask_refl_iso_conv_01s[i].values.astype(rasterio.int32) for i in range(len(mask_refl_iso_conv))]


# In[37]:


import rasterio
import shapely

shapes_iso_conv_cores = [rasterio.features.shapes(mask_refl_iso_conv_int[i]) for i in range(len(mask_refl_iso_conv))]


# In[38]:


polygons_iso_conv_cores = []

for i in range(len(mask_refl_iso_conv)):
    poly = [shapely.geometry.Polygon(shape[0]["coordinates"][0]) for shape in shapes_iso_conv_cores[i] if shape[1] == 1]
    polygons_iso_conv_cores.append(poly)


# #### get polygon coords

# In[39]:


x_coords = []
y_coords = []
times_all_iso_conv = []

for i in range(len(mask_refl_iso_conv)):
    for p in range(len(polygons_iso_conv_cores[i])):
        x,y = polygons_iso_conv_cores[i][p].exterior.coords.xy
        x_coords.append(np.array(x).astype(int))
        y_coords.append(np.array(y).astype(int))
        times_all_iso_conv.append(times_iso_conv_core[i])                                                  


# #### get lat/lon corresponding to polygon coords

# In[41]:


poly_lon = []

for c in range(len(x_coords)):
    lon_idx = x_coords[c]
    lon_idx[np.where(lon_idx==241)]=240
    lon_idx2 = np.where(lon_idx==2301, 2300, lon_idx)
    lon_vals = mask_refl_iso_conv[0].lon.values[lon_idx2]-360
    poly_lon.append(lon_vals)


# In[42]:


poly_lat = []

for c in range(len(y_coords)):
    lat_idx = y_coords[c]
    lat_idx2 = np.where(lat_idx==1201, 1200, lat_idx)
    lat_vals = mask_refl_iso_conv[0].lat.values[lat_idx2]
    poly_lat.append(lat_vals)


# In[43]:


from shapely.geometry import Polygon
polygon_geom_iso_conv = [Polygon(zip(poly_lon[c], poly_lat[c])) for c in range(len(y_coords))]


# #### turn into new polygons w/ lat/lon coords

# In[45]:


geo_idx_iso_conv = np.arange(0, len(times_all_iso_conv), 1)


# In[46]:


from geopandas import GeoSeries, GeoDataFrame

crs = {'init': 'epsg:4326'}
iso_conv_polygon = gp.GeoDataFrame(index=geo_idx_iso_conv, crs=crs, geometry=GeoSeries(polygon_geom_iso_conv))    
iso_conv_polygon['time'] = times_all_iso_conv


# ### test polygon

# ### designate region as having isolated convective rainfall if convective core intersects region

# In[4]:


print('find which regions isolated convective cores fall into-this takes several hours')


# In[56]:


iso_conv_core_region = []
iso_conv_core_date = []
iso_conv_core_count = []
iso_conv_core_id = []

for f in range(len(conus_reg)):
    for t in range(len(iso_conv_polygon)):
            if (conus_reg['geometry'].iloc[f].intersects(iso_conv_polygon['geometry'].iloc[t].buffer(0.02))):
                iso_conv_core_region.append(conus_reg['region_list'].iloc[f])
                iso_conv_core_date.append(iso_conv_polygon['time'].iloc[t])
                iso_conv_core_count.append(1)
                iso_conv_core_id.append(iso_conv_core_df2['object_idx'].iloc[t])
        


# ### make a df of isolate convective core regions

# In[57]:


iso_conv_core_region = pd.DataFrame({'region':iso_conv_core_region, 'time':iso_conv_core_date, 'occurrence':iso_conv_core_count, 'object_idx':iso_conv_core_id})


# #### turn time objects into datetimes

# In[58]:


iso_conv_core_region['time'] = [pd.to_datetime(iso_conv_core_region['time'].iloc[i], format='%Y%m%d'+'T'+'%H:%M:%S') for i in range(len(iso_conv_core_region))]
iso_conv_core_df2['time'] = [pd.to_datetime(iso_conv_core_df2['time'].iloc[i], format='%Y%m%d'+'T'+'%H:%M:%S') for i in range(len(iso_conv_core_df2))]


# In[59]:


iso_conv_core_region['time_int'] =iso_conv_core_region['time'].astype(int)/ 10**9
iso_conv_core_df2['time_int'] = iso_conv_core_df2['time'].astype(int)/ 10**9


# In[60]:


iso_conv_core_region_sort = iso_conv_core_region.sort_values(by='time')
iso_conv_core_df2_sort = iso_conv_core_df2.sort_values(by='time')


# ### merge this df with df of convective characteristics

# In[61]:


iso_conv_core_all = iso_conv_core_region_sort.merge(iso_conv_core_df2_sort, how='left', on=['object_idx', 'time_int'] )


# In[62]:


print(len(iso_conv_core_all))
#print(iso_conv_core_all[0:5])


# In[63]:


iso_conv_core_all = iso_conv_core_all.drop(['time_int','time_y'],axis=1)


# ### resample to daily resolution

# In[64]:


iso_conv_core_daily = iso_conv_core_all.groupby('region').resample('D', on='time_x').mean().reset_index()


# In[67]:


print('length of daily isolated convective cores')
print(len(iso_conv_core_daily))
#print(iso_conv_core_daily.iloc[0:3])


# In[66]:


iso_conv_core_daily.to_csv('isolated_convection_daily_bukovsky_regions_1995_1996.csv')


# In[ ]:




