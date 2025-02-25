#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:48:12 2023

@author: sandb425
"""
print("Regridded BHUWII Project")
print("Beginning imports.")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import xesmf as xe  # regridding


def S10_PP(radWm2, Tcelcius):
    C = 1159
    Ea = 0.283  # eV
    Tkelvin = Tcelcius + 273.15
    Popt = 836  # mg C m^-3 d^-1
    alpha = 7668  # m^2 photons^-1 x 10^-25
    Irrad = radWm2 * 24 * 60 * 60 / 3.61e-19 / 1e25 / 200
    # photons m^-2 x 10^25
    production = (
        C
        * np.exp(-Ea / 0.0000862 / Tkelvin)
        * Popt
        * (1 - np.exp(-alpha * Irrad / Popt))
    )
    return production  # mg C m^-3 d^-1


def is_leap_year(year):
    """if year is a leap year return True
    else return False"""
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0


def doy(Y, M, D):
    """given year, month, day return day of year
    Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7"""
    if is_leap_year(Y):
        K = 1
    else:
        K = 2
    N = int((275 * M) / 9.0) - K * int((M + 9) / 12.0) + D - 30
    return N


# %% Import files

glsea = xr.open_mfdataset("./glsea/20*_sst.nc")["sst"]  # SST from Great Lakes Environmental Surface Analysis
glsea_ice = xr.open_mfdataset("./glsea/20*_ice.nc")["ice_concentration"]  # Ice coverage from Great Lakes Environmental Surface Analysis
bath = xr.open_dataset("superior_lld.grd")  # Bathymetry and mask from NOAA
pfw_df = pd.read_csv("pfw_lowess.csv")  # Atmospheric CO2 from PFW/WLEF tall tower

# light https://psl.noaa.gov/data/gridded/data.narr.html
ds19 = xr.open_dataset("./Data/dswrf.2019.nc")["dswrf"]
ds20 = xr.open_dataset("./Data/dswrf.2020.nc")["dswrf"]
ds21 = xr.open_dataset("./Data/dswrf.2021.nc")["dswrf"]
ds22 = xr.open_dataset("./Data/dswrf.2022.nc")["dswrf"]
ds23 = xr.open_dataset("./Data/dswrf.2023.nc")["dswrf"]
ds_list = [ds19, ds20, ds21, ds22, ds23]
ds_rad = xr.concat(ds_list, dim="time")
for file in ds_list:
    file.close()

# u10 wind, also from NARR (see above)
uw19 = xr.open_dataset("./Data/uwnd.10m.2019.nc")["uwnd"]
uw20 = xr.open_dataset("./Data/uwnd.10m.2020.nc")["uwnd"]
uw21 = xr.open_dataset("./Data/uwnd.10m.2021.nc")["uwnd"]
uw22 = xr.open_dataset("./Data/uwnd.10m.2022.nc")["uwnd"]
uw23 = xr.open_dataset("./Data/uwnd.10m.2023.nc")["uwnd"]

vw19 = xr.open_dataset("./Data/vwnd.10m.2019.nc")["vwnd"]
vw20 = xr.open_dataset("./Data/vwnd.10m.2020.nc")["vwnd"]
vw21 = xr.open_dataset("./Data/vwnd.10m.2021.nc")["vwnd"]
vw22 = xr.open_dataset("./Data/vwnd.10m.2022.nc")["vwnd"]
vw23 = xr.open_dataset("./Data/vwnd.10m.2023.nc")["vwnd"]

vw_list = [vw19, vw20, vw21, vw22, vw23]
vw = xr.concat(vw_list, dim="time")
for file in vw_list:
    file.close()

uw_list = [uw19, uw20, uw21, uw22, uw23]
uw = xr.concat(uw_list, dim="time")
for file in uw_list:
    file.close()

wind = xr.merge([uw, vw])
wind = wind.assign(
    wspd=(wind.uwnd**2 + wind.vwnd**2) ** (0.5)
)  # / 1.94384  # kts to m/s
print("Files and packages imported.")

# %% Edit files

# define rectillinear bounds
min_lon = -92.12
min_lat = 46.4
max_lon = -84.34
max_lat = 49.01

# cut down bathymetry file to save time, space
bath = bath.sel(y=slice(min_lat, max_lat), x=slice(min_lon, max_lon))

# define rectillinear grid spacing at 0.03 degrees
lons_vector = np.linspace(
    bath.x.min().item(),
    bath.x.max().item(),
    num=int(np.round((bath.x.min().item() - bath.x.max().item()) * -50)) + 1,
)
lats_vector = np.linspace(
    bath.y.min().item(),
    bath.y.max().item(),
    num=int(np.round((bath.y.min().item() - bath.y.max().item()) * -50)) + 1,
)

# regrid and interpolate swr data
ds_rad = (
    ds_rad.where(ds_rad.lat > 44, drop=True)
    .where(ds_rad.lat < 51, drop=True)
    .where(ds_rad.lon < -81, drop=True)
    .where(ds_rad.lon > -95, drop=True)
)
# and wind
wind = (
    wind.where(wind.lat > 44, drop=True)
    .where(wind.lat < 51, drop=True)
    .where(wind.lon < -81, drop=True)
    .where(wind.lon > -95, drop=True)
)

glsea = (
    glsea.where(glsea.lat > 44, drop=True)
    .where(glsea.lat < 51, drop=True)
    .where(glsea.lon < -81, drop=True)
    .where(glsea.lon > -95, drop=True)
)

glsea_ice = (
    glsea_ice.where(glsea_ice.lat > 44, drop=True)
    .where(glsea_ice.lat < 51, drop=True)
    .where(glsea_ice.lon < -81, drop=True)
    .where(glsea_ice.lon > -95, drop=True)
)

mask = bath.interp(x=lons_vector).interp(y=lats_vector)
mask = xr.where(mask["z"] > 0, 0, 1)
mask.plot()

ds_shape_out = xr.Dataset(
    {
        "lat": (["lat"], lats_vector, {"units": "degrees_north"}),
        "lon": (["lon"], lons_vector, {"units": "degrees_east"}),
        "mask": (["lat", "lon"], mask.values),
    }
)
regridder = xe.Regridder(
    ds_rad, ds_shape_out, "bilinear", extrap_method="inverse_dist"
)
dr = ds_rad
ds_rad_out = regridder(dr)
ds_rad_out[1, :, :].plot(cmap=cmocean.cm.haline)

regridder = xe.Regridder(
    glsea, ds_shape_out, "bilinear", extrap_method="inverse_dist"
)
dr = glsea
glsea_out = regridder(dr)
glsea_out[1, :, :].plot(cmap=cmocean.cm.thermal)

regridder = xe.Regridder(
    glsea_ice, ds_shape_out, "bilinear", extrap_method="inverse_dist"
)
dr = glsea_ice
glsea_ice_out = regridder(dr)
glsea_ice_out[1, :, :].plot(cmap=cmocean.cm.ice)

regridder = xe.Regridder(
    wind, ds_shape_out, "bilinear", extrap_method="inverse_dist"
)
dr = wind.wspd
wind_out = regridder(dr)
wind_out[1, :, :].plot(cmap=cmocean.cm.thermal)
# %% Init params

lon = np.tile(lons_vector, (len(lats_vector), 1))  # 2D arrays of lon, lat
lat = np.flipud(np.tile(lats_vector, (len(lons_vector), 1)).T)
time = pd.date_range(
    "2019-01-01", "2023-12-31", freq="D"
)  # 1D time with daily res.
reference_time = pd.Timestamp("2006-01-01")

print("Prepared bathymetry and model bounds.")
# %% atmpCO2
atmpcotwos = np.empty(len(time))
for i in range(len(atmpcotwos)):
    ts = (time[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    atmospheric_xCO2 = pfw_df[np.abs(pfw_df.timestamp - ts) < 604800][
        "xCO2"
    ].mean()  # within a week
    this_atm = atmospheric_xCO2 * 99.1 / 101.325
    atmpcotwos[i] = this_atm
print("Prepared PFW atmospheric CO2 timeseries.")

# %% DOYs

time_to_doy = time.to_frame()
doy_list = time_to_doy.apply(
    lambda x: doy(
        pd.to_datetime(x[0]).year,
        pd.to_datetime(x[0]).month,
        pd.to_datetime(x[0]).day,
    ),
    axis=1,
)
# %% Construct and output netcdf
ds = xr.Dataset(
    data_vars=dict(
        pco=(
            ["x", "y", "time"],
            np.full([len(lons_vector), len(lats_vector), len(time)], np.nan),
        ),
        # depth=(
        #    ["y", "x"],
        #    bath["z"].interp(x=lons_vector).interp(y=lats_vector).values,
        # ),
        temp=(
            ["time", "y", "x"],
            glsea_out.sel(time=time.to_list(), method="nearest").values,
        ),
        airsp=(
            ["time", "y", "x"],
            wind_out.sel(time=time.to_list(), method="nearest").values,
        ),
        # flux=(
        #    ["x", "y", "time"],
        #    np.full([len(lons_vector), len(lats_vector), len(time)], np.nan),
        # ),
        light=(
            ["time", "y", "x"],
            ds_rad_out.sel(time=time.to_list(), method="nearest").values,
        ),
        primprod=(
            ["time", "y", "x"],
            S10_PP(
                ds_rad_out.sel(time=time.to_list(), method="nearest").values,
                glsea_out.sel(time=time.to_list(), method="nearest").values,
            ),
        ),
        atmpco=(
            ["time"],
            atmpcotwos,
        ),
        ice=(
            ["time", "y", "x"],
            glsea_ice_out.sel(time=time.to_list(), method="nearest").values,
        ),
        doy=(["time"], doy_list),
    ),
    coords=dict(
        # lon=(["y", "x"], lon),
        # lat=(["y", "x"], lat),
        x=lons_vector,
        y=lats_vector,
        time=time,
        reference_time=reference_time,
    ),
    attrs=dict(
        description="Parameters used for FFNN prediction of pCO2 over Lake Superior"
    ),
)
print("Assembled BHUWII input file.")
ds.to_netcdf("bhuw_input.nc")
print("Saved BHUWII input file.")


# %% underway data prep
#grab Blue Heron underway data file
BH = pd.read_csv("BH1923_Processed.csv")

BH["rad"] = np.nan

for i in BH.index: #tack on matching PAR as above
    this_lon = BH.loc[i, "lon"]
    this_lat = BH.loc[i, "lat"]
    this_datetime = BH.loc[i, "Date"]
    this_rad = (
        ds_rad_out.interp(lon=this_lon, lat=this_lat)
        .sel(time=this_datetime, method="nearest")
        .item()
    )
    BH.loc[i, "rad"] = this_rad

BH.to_csv("BH1923_Processed_Rad.csv", index=False)
