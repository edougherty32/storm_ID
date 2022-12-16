#!/usr/bin/env python
# coding: utf-8

# ### script for outputting 850 hPa theta_e from ERA5 from 1995–2017
# ### date created: 6 July 2021
# ### author: Erin Dougherty (doughert@ucar.edu)

# In[34]:


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
from dask.distributed import Client
#from wrf import getvar, ALL_TIMES


# In[35]:

from dask_jobqueue import PBSCluster

cluster = PBSCluster(cores=6,
                     memory="200GB",
                     project='P54048000',
                     queue='regular',
                     walltime='12:00:00')

cluster.scale(10)  # Start 100 workers in 100 jobs that match the description above

from dask.distributed import Client
client = Client(cluster)    # Connect to that cluster


# In[36]:


# ### set bounds for ERA5 files

# In[38]:


rlon = 115+180
llon= 55+180 
llat = 25
ulat = 49


# ### open all ERA5 files between 1995–2017 at 850 hPa level 

# In[39]:
# INPUT YEAR OF INTEREST HERE
year_nm = 1995
years = np.linspace(year_nm,year_nm,num=1, dtype=int)
year_str = years.astype(str)


# ### avoid creating large chunks

# In[40]:


import dask
dask.config.set({"array.slicing.split_large_chunks": True})


# ### import T

# In[41]:


import os.path
fflood_T_pl = []

for c, item in enumerate(year_str): 
    print(c)
    fpath = glob.glob(os.path.join('/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'+item+'*/e5.oper.an.pl.128_130_t.ll025sc.*.nc'))
    file_match = xr.open_mfdataset(fpath, lock=False, parallel=True).sel(level=850)
    file_pl_T = file_match['T'].where((rlon >= file_match.longitude) & 
                                              (llon <= file_match.longitude) &
                                              (llat <= file_match.latitude) &  
                                              (ulat >= file_match.latitude), drop=True)
    file_match.close()
    fflood_T_pl.append(file_pl_T)


# In[42]:


print('T imported')


# ### import Q

# In[43]:


fflood_Q_pl = []

for c, item in enumerate(year_str): 
    print(c)
    fpath = glob.glob(os.path.join('/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'+item+'*/e5.oper.an.pl.128_133_q.ll025sc.*.nc'))
    file_match = xr.open_mfdataset(fpath, lock=False, parallel=True).sel(level=850)
    file_pl_Q = file_match['Q'].where((rlon >= file_match.longitude) & 
                                              (llon <= file_match.longitude) &
                                              (llat <= file_match.latitude) &  
                                              (ulat >= file_match.latitude), drop=True)
    file_match.close()
    fflood_Q_pl.append(file_pl_Q)


# In[44]:


print('Q imported')


# ### import RH

# In[45]:


import os.path

fflood_RH_pl = []

for c, item in enumerate(year_str):   
    print(c)
    fpath = glob.glob(os.path.join('/gpfs/fs1/collections/rda/data/ds633.0/e5.oper.an.pl/'+item+'*/e5.oper.an.pl.128_157_r.ll025sc.*.nc'))
    file_match = xr.open_mfdataset(fpath, lock=False, parallel=True).sel(level=850)
    file_pl_RH = file_match['R'].where((rlon >= file_match.longitude) & 
                                              (llon <= file_match.longitude) &
                                              (llat <= file_match.latitude) &  
                                              (ulat >= file_match.latitude), drop=True)
    file_match.close()
    fflood_RH_pl.append(file_pl_RH)


# In[46]:


print('RH imported')


# #### calculate dewpoint (in K)

# In[47]:


fflood_Td_pl = [fflood_T_pl[i] - ((100-fflood_RH_pl[i])/5) for i in range(len(fflood_T_pl))]


# In[48]:


print('dewpoint calculated')


# ### compute theta_e following:
# #### https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.equivalent_potential_temperature.html
# #### https://journals.ametsoc.org/view/journals/mwre/108/7/1520-0493_1980_108_1046_tcoept_2_0_co_2.xml?tab_body=pdf 

# #### 1) compute temperature of LCL

# In[49]:


Tl = [ (1/ ( (1/(fflood_Td_pl[i]-56)) + ((np.log(fflood_T_pl[i]/fflood_Td_pl[i]))/800) )) +56 for i in range(len(fflood_T_pl))]


# In[50]:


print('LCL temp computed')


# #### 2) compute the mixing ratio

# In[51]:


r = [fflood_Q_pl[i]/(1-fflood_Q_pl[i]) for i in range(len(fflood_T_pl))]


# In[52]:


print('mixing ratio calculated')


# #### 3) compute the vapor pressure (e)

# In[53]:


K = 0.286
p = 85000 # pressure in Pa

e = [(8500*r[i])/(622+r[i]) for i in range(len(fflood_T_pl))]


# In[54]:


print('vapor pressure calculated')


# #### 4) compute potential temperature at LCL

# In[55]:


theta_dl = [fflood_T_pl[i]*(100000/(p-e[i]))**K *(fflood_T_pl[i]/Tl[i])**(0.28*r[i]) for i in range(len(fflood_T_pl))]


# #### 5) compute theta_e

# In[56]:


thetae = [theta_dl[i]*np.exp( ((3036/Tl[i])-1.78) * r[i]*(1+0.448*r[i])) for i in range(len(fflood_T_pl))]


# In[57]:


print('theta_e calculated')


# ### rename variable in theta_e dataset

# In[58]:


for i in range(len(fflood_T_pl)):
    thetae[i].name = 'theta_e'


# #### turn each data array into a data set

# In[28]:


thetae_ds = [thetae[i].to_dataset() for i in range(len(fflood_T_pl))]


# #### resample to 6 hourly means

# In[29]:


print('resampling data to 6 hour mean')


# In[30]:


thetae_resample = [thetae_ds[i].resample(time='6H').mean() for i in range(len(thetae_ds))]


# In[150]:


print('resampling complete')


# ### apply gaussian filter


# In[151]:


print('apply gaussian filter- this could take awhile')


# In[93]:


from scipy.ndimage import gaussian_filter

thetae_sm = []
 
for i in range(0,1460):
    print(i)
    smooth = gaussian_filter(thetae_resample[0]['theta_e'][i], sigma=5)
    thetae_sm.append(smooth)


# In[152]:


print('smoothing complete! turning into xarray dataset')


# ### turn list of arrays into xarray dataset

# In[129]:


thetae_sm_xr = xr.Dataset(data_vars=dict(
                                thetae = (['time', 'latitude', 'longitude'], thetae_sm)),
                            coords=dict(
                                longitude=('longitude', thetae_resample[0]['theta_e'].longitude.values), 
                                latitude=('latitude', thetae_resample[0]['theta_e'].latitude.values),
                                time=thetae_resample[0]['theta_e'].time), 
                             attrs=dict(
                                 description='850 hPa theta_e', 
                             units = 'K',)
                           )


# In[153]:


print('exporting to netcdf')


# ### export to netcdf 

# In[140]:


thetae_sm_xr.to_netcdf('era5_thetae_%s.nc' %str(year_nm))


# In[154]:


print('exported')


