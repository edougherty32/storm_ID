# storm_ID
## Code to identify daily storm types across the CONUS 
![storm_type_algorithm](https://user-images.githubusercontent.com/40123935/208155319-964df847-b371-4818-8bd8-1e71635e4235.png)

# Overview:
This code uses data from 1995–2017 (due to limitations of radar availability) to classify storm types in the continental U.S. by their occurrence in the Bukovsky regions and relates them to local and regional floods. Storm types are first categorized by synoptic type–tropical cyclone (TC), extratropical cyclone (ETC), or frontal, then by their radar attributes–mesoscale convective system (MCS), isolated convection, or stratiform rainfall. Please see Fig. 2 (picutred above) in Brunner and Dougherty (2022) for the decision tree used to determine the predominant storm type on each day. 

# Code structure
Each folder provides the code necessary to identify each storm type (TC, ETC, fronts, MCS, isolated convection, and non-convective/stratiform rainfall). Further descriptions of each algorithm for storm type are provided:

## Fronts: 
- The 850 hPa theta_e gradient is calculated from ERA5 (available on Cheyenne) to determine if a front exists based on the definition from Sprenger et al. (2017) using era5_thetae_calc.py. Given how slow this takes to run, run this script for each year of interest using a bash script to submit this to the supercomputer (run_thetae_calc_ch.csh). 
- Use frontx_ID_bukovsky.ipynb and fronty_ID_bukovsky.ipynb to determine if theta_e gradient exceeds 3.6 K/100 km and locate where the fronts fall into the Bukovsky regions. 
- Use front_analysis_bukovsky_regions.ipynb to import x-direction and y-direction fronts 

## ETCs:
- Uses cyclone climatology from ETH Zurich that identifies hourly cyclone tracks and minimum pressure from ERA5
  - Wernli, H., & Schwierz, C. (2006). Surface Cyclones in the ERA-40 Dataset (1958–2001). Part I: Novel Identification Method and Global Climatology, Journal of the Atmospheric Sciences, 63(10), 2486-2507
  - Sprenger, M., Fragkoulidis, G., Binder, H., Croci-Maspoli, M., Graf, P., Grams, C. M., Knippertz, P., Madonna, E., Schemm, S., Škerlak, B., & Wernli, H. (2017). Global Climatologies of Eulerian and Lagrangian Flow Features based on ERA-Interim, Bulletin of the American Meteorological Society, 98(8), 1739-1748
- Requested from personal correspondence with Henri Wernil and Michael Sprenger
- Identified in each region if ETC center falls within region in ETC_id_buvosky.ipynb

## TCs
Used [Extended Best Track](https://rammb2.cira.colostate.edu/research/tropical-cyclones/tc_extended_best_track_dataset/) dataset to identify if 34-kt wind radii of TC overlaps with Bukovsky regions 

## Radar identification (MCS, isolated convection, and stratiform rainfall)
Run filter_radar_years.ipynb first to filter and declutter [GridRad reflectivity data](http://gridrad.org/data.html) from 1995–2017
Outputs filtered radar data for every 2 years (1995–1997, 1997–1999, 2000–2002, 2003–2005, 2006–2008, 2009–2011, 2012–2014, 2015–2017) since so large. 

### MCS
- Uses filtered radar data to identify MCSs if > 40 dBZ over at least 100 km in one direction in MCS_classification_py.py
- This takes a long time, so python script is submitted to Cheyenne queue using run_mcs.sh
- Rerun for each 2-year period
- Merge all 2-year period files in merge_MCS_classifications.ipynb

### Isolated convection
- Uses filtered radar data to identify isolated convection if > 40 dBZ in isolated_convective_classification_py.py
- This takes a long time, so python script is submitted to Cheyenne queue using run_iso.sh
- Rerun for each 2-year period
- Merge all 2-year period files in merge_iso_conv_classifications.ipynb

### Non-convective/stratiform
- Uses filtered radar data to identify stratiform rainfall  if > 18 dBZ and < 40 dBZ in non_convect_classification_py.py
- This takes a long time, so python script is submitted to Cheyenne queue using run_non_convect.sh
- Rerun for each 2-year period
- Merge all 2-year period files in merge_non_conv_classifications.ipynb

## Final step:
- Merge all the dataframes of storm types into one dataframe in merge_storm_types.ipynb
This uses decision tree from Fig. 2 in Brunner and Dougherty (2022) to decide predominant cause of rainfall 

