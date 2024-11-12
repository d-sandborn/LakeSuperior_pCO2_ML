#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:29:19 2023

@author: sandb425
"""
import numpy as np  # For numerical fast numerical calculations
import matplotlib.pyplot as plt  # For making plots
import pandas as pd  # Deals with data
import seaborn as sns  # Makes beautiful plots
from sklearn.preprocessing import StandardScaler  # Testing sklearn
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import xarray as xr
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import crs
import matplotlib
import pyproj as proj
import cmocean
import itertools
from matplotlib.patheffects import Stroke, Normal
import cartopy.mpl.geoaxes
import pyseaflux as sf  # flux calc harmonization
import PyCO2SYS as pyco2
import scienceplots
import gsw
from statsmodels.tsa.seasonal import MSTL
import math
import statsmodels.api as sm

plt.style.use(
    [
        "science",
    ]
)  # "no-latex"


model_output_file = "regression_output.nc"
ds = xr.open_dataset(model_output_file)


from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:5609", "EPSG:4326")
transformer.transform(-92, 47)

# %% Define


def flux_calculation(ds, coef=0.251):
    """
    DEPRECATED

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    coef : TYPE, optional
        DESCRIPTION. The default is 0.251.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    ds["flux"] = (
        ("nx", "ny", "time"),
        np.empty([61, 30, len(ds.time.values)]),
    )
    for i in ds.nx.values:
        for j in ds.ny.values:
            for t in ds.time.values:
                wind = 1  # dummy
                Sc = 600  # dummy
                this_flux = coef * wind**2 + (Sc / 600) ** (-0.5)
                ds["flux"][i, j, t] = this_flux
    return ds


def plot_map(
    file,
    time=0,
    variable="pCO2_pred",
    axis_lims=(-92.3, -84.3, 46.3, 49.1),
    plot_size=(6.4, 4.8),
    show_grid=False,
    title=None,
    logscale=False,
    color_bar=True,
    var="count",
    vmin=None,
    vmax=None,
    cmap="viridis",
    heading=None,
    dpi=500,
    credit=None,
    plot_file_name=None,
    plot_bathy=False,
    translate=False,
):
    """Plot (and-or save) a heatmap from a saved netCDF."""
    ds = xr.open_dataset(file, engine="netcdf4")
    this_datetime = str(ds["time"].isel(time=time).values)[0:16]
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["font.size"] = 8
    fig = plt.figure(figsize=(plot_size[0], plot_size[1]))
    ax = fig.add_subplot(
        projection=ccrs.AlbersEqualArea(
            central_longitude=-88, central_latitude=47
        ),
    )

    ax.set_extent(axis_lims, crs=ccrs.PlateCarree())

    if plot_bathy is True:
        bath = xr.open_dataset("superior_lld.grd")
        X, Y = np.meshgrid(bath.x.values, bath.y.values)
        contouring = ax.contour(
            X,
            Y,
            -bath.where(bath.z < 0).z.values,
            levels=[100, 200, 300, 400],
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            colors="k",
            linestyles="-",
            zorder=4,
        )
        ax.clabel(contouring, inline=True, fontsize=5)

    ax.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=1)
    # ax.add_feature(
    #    cfeature.LAKES, zorder=2, facecolor="white", edgecolor="black"
    # )
    ax.add_feature(
        cfeature.RIVERS, zorder=2, facecolor="dimgrey", edgecolor="lightblue"
    )
    # ax.add_feature(cfeature.STATES, ls="-.", edgecolor="black", alpha=0.5)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "rivers_north_america", "10m"
        ),
        facecolor="dimgrey",
        edgecolor="lightblue",
    )
    # ax.add_feature(
    #    cfeature.NaturalEarthFeature("physical", "lakes_north_america", "10m"),
    #    edgecolor="black",
    #    facecolor="white",
    #    zorder=2,
    # )
    X = ds.x.values  # - 0.05
    Y = ds.y.values  # + 0.05
    if translate:
        C = ds[variable].isel(time=time).values.T
    else:
        C = ds[variable].isel(time=time).values
    pc = ax.pcolormesh(
        X,
        Y,
        C,
        cmap=cmap,
        shading="gouraud",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    pc.set_clim(vmin, vmax)

    # ax.set_xticks(np.arange(-92.5, -83, 1))
    # ax.set_yticks(np.arange(46, 49.5, 0.5))
    ax.set_axis_off()
    gl = ax.gridlines(
        draw_labels=True,
        ls=":",
        alpha=0.5,
        xlocs=np.arange(-92, -82, 2),
        ylocs=np.arange(46, 50, 1),
    )
    gl.right_labels = False
    gl.top_labels = False
    # ax.tick_params(axis="both", direction="in", right=True, top=True)
    # ax.zebra_frame(lw = 5, crs = ccrs.PlateCarree(), zorder = 4)

    if title is not None:
        ax.set_title(title + " at " + this_datetime)
    if color_bar:
        plt.colorbar(pc, ax=ax, shrink=0.5, pad=0.01)
    if credit is not None:
        ax.annotate(
            "DES 2023",
            xy=(0.8, 0.05),
            xycoords="axes fraction",
            fontsize=5,
            fontstyle="italic",
        )

    # if axis_lims is not None:
    #    ax.set_extent(axis_lims, transform = ccrs.PlateCarree())
    fig.tight_layout()
    ds.close()

    plt.savefig("./Plots/" + variable + ".png")


def timeseries_at_coord(ds, lon, lat, variable, fmt="k-"):
    """
    Simple timeseries plotting.
    """
    this_nx = np.argmin(np.abs(ds.lon.values[0, :] - lon))
    this_ny = np.argmin(np.abs(ds.lat.values[:, 0] - lat))
    fig, ax = plt.subplots()
    ds[variable].isel(nx=this_nx, ny=this_ny).plot.line(fmt, ax=ax)
    ax.set_title(variable + " at " + str(lat) + ", " + str(lon))
    plt.tight_layout()


def zebra_frame(self, lw=2, crs=None, zorder=None):
    # Alternate black and white line segments
    bws = itertools.cycle(["k", "w"])

    self.spines["geo"].set_visible(False)

    left, right, bottom, top = self.get_extent()

    # xticks = sorted([left, *self.get_xticks(), right])
    xticks = sorted([*self.get_xticks()])
    xticks = np.unique(np.array(xticks))
    # yticks = sorted([bottom, *self.get_yticks(), top])
    yticks = sorted([*self.get_yticks()])
    yticks = np.unique(np.array(yticks))

    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[yticks[0], yticks[0]], [yticks[-1], yticks[-1]]]
            else:
                xs = [[xticks[0], xticks[0]], [xticks[-1], xticks[-1]]]
                ys = [[start, end], [start, end]]

            # For first and last lines, used the "projecting" effect
            capstyle = (
                "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            )
            for xx, yy in zip(xs, ys):
                self.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=max(
                        0, lw - self.spines["geo"].get_linewidth() * 2
                    ),
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        Stroke(linewidth=lw, foreground="black"),
                        Normal(),
                    ],
                )


# setattr(cartopy.mpl.geoaxes.GeoAxes, "zebra_frame", zebra_frame)


def flux_calc(pco2w, pco2atm, u10, sst, sal):
    Sc = (
        1923.6
        - 125.06 * sst
        + 4.3773 * sst**2
        - 0.085681 * sst**3
        + 0.00070284 * sst**4
    )
    # should this be 600 or 660? Need to check windspeed units.  cm/hr
    littlek = (
        0.266 * (u10 * 0.514444) ** 2 * (Sc / 600) ** (-0.5)
    )  # knots to m/s, littlek is cm/hr
    Ko = np.exp(
        -58.0931
        + 90.5069 * (100 / (273.15 + sst))
        + 22.2940 * np.log((sst + 273.15) / 100)
        + sal
        * (
            0.027766
            - 0.025888 * (273.15 + sst) / 100
            + 0.0050578 * ((sst + 273.15) / 100) ** 2
        )
    )  # mols/L/atm
    # cm/hr * mol/L/atm * μatm * atm/10^6 μatm * 1000 L/m^3 / 100 cm/m * 1000 mmol/mol * 24 hr/day = mmol/m^2/day
    flux = littlek * Ko * (pco2w - pco2atm) / 1e6 * 1000 / 100 * 1000 * 24
    return flux


def pyflux_calc(ds, ice_corr=False):
    ds["area"] = (
        ["x", "y"],
        sf.area.area_grid(ds.y.values, ds.x.values, return_dataarray=False),
    )
    ds["kw"] = (
        ["time", "y", "x"],
        sf.gas_transfer_velocity.k_Ho06(ds.airsp.values**2, ds.temp.values)
        * (24 / 100),  # cm/hr --> m/d
    )
    ds[
        "K0"
    ] = (  # solubility of CO2 in seawater in mol/L/atm -- > mol/m^3/μatm
        ["time", "y", "x"],
        sf.solubility.solubility_weiss1974(
            0.05, ds.temp + 273.15, press_atm=0.98, checks=True
        ).data
        * (1e3 / 1e6),
    )
    if ice_corr:
        R_ice = (
            1 - ds.ice.where(ds.ice < 100.01).where(ds.ice > -0.001) / 100
        ).fillna(1)
    else:
        R_ice = 1
    ds["flux_mol_m2_day"] = (
        ["time", "y", "x"],
        (ds.kw * ds.K0 * (ds.pco - ds.atmpco) * R_ice).data,
    )
    ds["flux_avg_yr"] = (
        ["y", "x"],
        (ds.flux_mol_m2_day.mean("time") * 365).data,
    )  # molC/m2/year
    flux_integrated = (
        (ds.flux_mol_m2_day * ds.area * 12.011)
        .sum(dim=["y", "x", "time"])
        .values
        / 1e12
        / 5
        # * 365 #?
    )  # TgC/year
    return flux_integrated


def abline(slope, intercept, **kwargs):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "-", **kwargs)


class GridShader:
    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="x")
        self.cid = self.ax.callbacks.connect("xlim_changed", self.shade)
        self.shade()

    def clear(self):
        for span in self.spans:
            try:
                span.remove()
            except:
                pass

    def shade(self, evt=None):
        self.clear()
        xticks = self.ax.get_xticks()
        xlim = self.ax.get_xlim()
        xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[-1])]
        locs = np.concatenate(([[xlim[0]], xticks, [xlim[-1]]]))

        start = locs[1 - int(self.sf) :: 2]
        end = locs[2 - int(self.sf) :: 2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axvspan(s, e, zorder=0, **self.kw))


def flux_monte_carlo(ds, ice_corr=False, iterations=10, flux_var=30):
    flux_products = np.empty(iterations)
    for i in range(iterations):
        altered_ds = ds.copy()
        altered_pco = np.random.normal(
            loc=0, scale=flux_var, size=ds.pco.shape
        )
        altered_ds["pco"] = altered_ds["pco"] + altered_pco
        ds_refluxed, flux_integrated = pyflux_calc(
            altered_ds, ice_corr=ice_corr
        )
        flux_products[i] = flux_integrated
        return flux_products


# %% Animation
def pCO2_animator(ds):
    # Writer = animation.writers["pillow"]
    # writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    vmin = 200
    vmax = 500
    plt.rcParams["figure.dpi"] = 100
    subplot_kw = {"projection": ccrs.Orthographic(-88.5, 47.5)}
    fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(7, 5))
    ax.add_feature(cfeature.LAKES, color="blue", alpha=0.05)
    # ax.add_feature(cfeature.BORDERS, ls = ':')
    ax.add_feature(cfeature.STATES, ls="-", edgecolor="grey", alpha=0.3)
    gl = ax.gridlines(draw_labels=True, alpha=0.5)
    gl.right_labels = False
    gl.top_labels = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", direction="in", right=True, top=True)
    ax.set_title("$p$CO$_2$ ($\mu$atm)")
    X = ds.x.values  # [0, 0, :] - 0.05
    Y = ds.y.values  # [0, :, 0] + 0.05
    C = ds["pco"].isel(time=0).values.T
    pc = ax.pcolormesh(
        X,
        Y,
        C,
        cmap=cmocean.cm.curl,
        shading="gouraud",
        transform=ccrs.PlateCarree(),
    )
    pc.set_clim(vmin, vmax)
    plt.colorbar(pc, ax=ax, shrink=0.5, pad=0.05)

    def update(frame):
        X = ds.x.values  # [frame, 0, :] - 0.05
        Y = ds.y.values  # [frame, :, 0] + 0.05
        C = ds["pco"].isel(time=frame).values.T
        pc = ax.pcolormesh(
            X,
            Y,
            C,
            cmap=cmocean.cm.curl,
            shading="gouraud",
            transform=ccrs.PlateCarree(),
        )
        pc.set_clim(vmin, vmax)
        return pc

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=ds.time.values.shape[0] - 1, interval=30
    )
    plt.show()
    ani.save("test_animation.gif", writer="pillow")


# pCO2_animator(ds)

# %% Total flux calc
"""
flux_integrated = pyflux_calc(ds, ice_corr=True)  # GgC/year

print(
    "Total CO2 Efflux over "
    + str(ds.time[0].values)
    + " to "
    + str(ds.time[-1].values)
    + ": \n"
    + str(np.round(flux_integrated, 5))
    + " TgC/yr"
)

flux_integrated = pyflux_calc(
    ds.where(ds.doy > 99).where(ds.doy < 301), ice_corr=True
)  # GgC/year

print(
    "Total CO2 Efflux over "
    + "DOY 100"
    + " to "
    + "DOY 300"
    + ": \n"
    + str(np.round(flux_integrated, 5))
    + " TgC/yr"
)

flux_integrated = pyflux_calc(
    ds.where(ds.doy < 100), ice_corr=True
)  # GgC/year

print(
    "Total CO2 Efflux over "
    + "DOY 1"
    + " to "
    + "DOY 99"
    + ": \n"
    + str(np.round(flux_integrated, 5))
    + " TgC/yr"
)

flux_integrated = pyflux_calc(
    ds.where(ds.doy > 300), ice_corr=True
)  # GgC/year

print(
    "Total CO2 Efflux over "
    + "DOY 301"
    + " to "
    + "DOY 365"
    + ": \n"
    + str(np.round(flux_integrated, 5))
    + " TgC/yr"
)

# mc_flux = flux_monte_carlo(ds, ice_corr=True, iterations=10, flux_var=30)

flux_integrated_19 = (
    pyflux_calc(ds.sel(time=slice("2019-01-01", "2019-12-31")), ice_corr=True)
    * 5
)

flux_integrated_20 = (
    pyflux_calc(ds.sel(time=slice("2020-01-01", "2020-12-31")), ice_corr=True)
    * 5
)

flux_integrated_21 = (
    pyflux_calc(ds.sel(time=slice("2021-01-01", "2021-12-31")), ice_corr=True)
    * 5
)

flux_integrated_22 = (
    pyflux_calc(ds.sel(time=slice("2022-01-01", "2022-12-31")), ice_corr=True)
    * 5
)

flux_integrated_23 = (
    pyflux_calc(ds.sel(time=slice("2023-01-01", "2023-12-31")), ice_corr=True)
    * 5
)
"""
# %% Mapping
upper_right = (-84, 49)
lower_left = (-93, 46)

lims = (
    lower_left[0],  # lower left x
    upper_right[0],  # upper right x
    lower_left[1],  # lower left y
    upper_right[1],
)  # upper right y

plot_map(
    file=model_output_file,
    time=200,
    variable="pco",
    # axis_lims=lims,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title="Predicted $p$CO$_2$ ($\mu$atm)",
    logscale=False,
    color_bar=True,
    var="count",
    vmin=200,
    vmax=550,
    cmap=cmocean.cm.curl,
    heading=None,
    dpi=100,
    credit=True,
    plot_file_name="map_pco2.png",
    plot_bathy=False,
    translate=True,
)
# %% More Maps
plot_map(
    file=model_output_file,
    time=200,
    variable="primprod",
    # axis_lims=lims,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title="Predicted Prim. Prod.",
    logscale=False,
    color_bar=True,
    var="count",
    vmin=None,
    vmax=None,
    cmap=cmocean.cm.algae,
    heading=None,
    dpi=100,
    plot_file_name="map_primprod.png",
)

plot_map(
    file=model_output_file,
    time=200,
    variable="light",
    # axis_lims=lims,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title="Irradiation",
    logscale=False,
    color_bar=True,
    var="count",
    vmin=None,
    vmax=None,
    cmap=cmocean.cm.solar,
    heading=None,
    dpi=100,
    plot_file_name="map_light.png",
)

plot_map(
    file=model_output_file,
    time=200,
    variable="airsp",
    # axis_lims=lims,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title="Windspeed (m s$^{-1}$)",
    logscale=False,
    color_bar=True,
    var="count",
    vmin=None,
    vmax=None,
    cmap=cmocean.cm.dense,
    heading=None,
    dpi=100,
    credit=True,
    plot_file_name="map_airsp.png",
)

plot_map(
    file=model_output_file,
    time=200,
    variable="temp",
    # axis_lims=lims,
    plot_size=(6.4, 4.8),
    show_grid=False,
    title="SST",
    logscale=False,
    color_bar=True,
    var="count",
    vmin=None,
    vmax=None,
    cmap=cmocean.cm.thermal,
    heading=None,
    dpi=100,
    credit=True,
    plot_file_name="map_temp.png",
)


# %% Old Bathymetry
"""
file = model_output_file
time = -1
variable = "depth"
# axis_lims=lims
plot_size = (6.4, 4.8)
show_grid = False
title = "Bathymetry (m)"
logscale = False
color_bar = True
var = "count"
vmin = 0
vmax = -406
cmap = cmocean.cm.deep_r
heading = None
dpi = 300
plot_file_name = "map_depth.png"
this_datetime = str(ds["time"].isel(time=time).values)[0:16]
plt.rcParams["figure.dpi"] = dpi
subplot_kw = {"projection": ccrs.Orthographic(-88.5, 47.5)}
fig, ax = plt.subplots(
    subplot_kw=subplot_kw, figsize=(plot_size[0], plot_size[1])
)

# ax.set_extent(axis_lims, crs=crs.Orthographic(-92, 47))
X = ds.x.values
Y = ds.y.values
C = ds[variable].where(ds[variable] < 0).values
pc = ax.pcolormesh(
    X,
    Y,
    C,
    cmap=cmap,
    shading="gouraud",
    transform=ccrs.PlateCarree(),
)

# if axis_lims is None:
#    axis_lims = [
#        x[0],
#        x[-1],
#        y[0],
#        y[-1],
#    ]  # set axis limits to those of the grid

# ax.add_feature(cfeature.LAKES, color="blue", alpha=0.05)
# ax.add_feature(cfeature.BORDERS, ls = ':')
ax.add_feature(cfeature.STATES, ls="-", edgecolor="grey", alpha=0.3)
gl = ax.gridlines(draw_labels=True, alpha=0.5)
gl.right_labels = False
gl.top_labels = False
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis="both", direction="in", right=True, top=True)
if title is not None:
    ax.set_title(title)

pc.set_clim(vmin, vmax)
if color_bar:
    plt.colorbar(pc, ax=ax, shrink=0.5, pad=0.05)
ax.annotate(
    "DES 2023",
    xy=(0.8, 0.05),
    xycoords="axes fraction",
    fontsize=5,
    fontstyle="italic",
)
fig.tight_layout()
ds.close()
"""
# %% Timeseries

# timeseries_at_coord(ds, -89.5, 47.4, "pred_pCO2")
mcqs = pd.read_csv("PrelimBuoyData.csv", header=1, parse_dates=["Time"])
mcqw = pd.read_csv("Data/winter_mooring.csv", parse_dates=["Time"])
mcq = pd.merge(mcqs, mcqw, how="outer")
mcq_daily = mcq.set_index("Time")
mcq_daily = mcq_daily.resample("D").mean()
mcq_daily = mcq_daily.reset_index(drop=False)

# mcq["Time"] = pd.to_datetime(mcq["Time"])

fig, ax = plt.subplots()
atm = ds["atmpco"].plot.line("b-", ax=ax, label="Atmospheric (WLEF)")
ax.set_title("Atmospheric pCO$_2$")
ax.set_ylabel("$p$CO$_2$ ($\mu$atm)")
plt.legend()
plt.tight_layout()
plt.savefig("./Plots/atm_timeseries.png")

fig, ax = plt.subplots(figsize=(12, 3))
atm = ds["atmpco"].plot.line("b-", ax=ax, label="Atmospheric (WLEF)")

sns.lineplot(x=mcq.Time, y=mcq.pCO2, label="Mooring @ 10 m", c="r", ax=ax)
water = (
    ds["pco"]
    .isel(
        x=np.argmin(np.abs(ds.x.values + 91.57)),
        y=np.argmin(np.abs(ds.y.values - 46.98)),
    )
    .plot.line("k-", ax=ax, label="Surface Model")
)
ax.set_title("Atmospheric and Sea Surface pCO$_2$ c. McQuade Offshore")
ax.set_ylabel("$p$CO$_2$ ($\mu$atm)")
plt.legend()
plt.tight_layout()
plt.savefig("./Plots/mcq_long_timeseries.png")

# 48.06112538369662, -87.79306632829996
# timeseries_at_coord(ds, -89.5, 47.4, "pred_pCO2")
"""
fig, ax = plt.subplots(figsize=(12, 3))
atm = ds["atmpco"].plot.line("b-", ax=ax, label="Atmospheric (WLEF)")
water = (
    ds["pco"]
    .isel(
        x=np.argmin(np.abs(ds.x.values + 87.79306632829996)),
        y=np.argmin(np.abs(ds.y.values - 48.06112538369662)),
    )
    .plot.line("k-", ax=ax, label="Surface Model")
)
ax.set_title("Atmospheric and Sea Surface pCO$_2$ c. Central Mooring")
ax.set_ylabel("pCO$_2$ ($\mu$atm)")
plt.legend()
plt.tight_layout()
"""

# %% Mooring Validation

sns.set(rc={"figure.figsize": (6, 6)})
sns.set_style("ticks")
sns.despine()
fig, ax = plt.subplots(dpi=200)
mooring = ax.plot(
    mcq_daily.Time, mcq_daily.pCO2, label="Mooring $p$CO$_2$", c="r"
)
ax2 = ax.twinx()
mooringtemp1 = ax2.plot(
    mcq.Time, mcq.TC, c="r", ls=":", label="Mooring Temperature"
)
mooringtemp2 = ax2.plot(mcq.Time, mcq["Temperature.1"], c="r", ls=":")
sns.lineplot(
    x=ds.time,
    y=ds["pco"].isel(
        x=np.argmin(np.abs(ds.x.values + 91.84)),
        y=np.argmin(np.abs(ds.y.values - 46.81)),
    ),
    label="Model $p$CO$_2$",
    c="k",
    ax=ax,
)
sns.lineplot(
    x=ds.time,
    y=ds["temp"].isel(
        x=np.argmin(np.abs(ds.x.values + 91.84)),
        y=np.argmin(np.abs(ds.y.values - 46.81)),
    ),
    c="k",
    ls=":",
    label="GLSEA Temperature",
    ax=ax2,
)
ax.set_xlim([19247.222569444446, 19520.99270833333])
ax.set_ylim(250, 550)
ax.tick_params(
    axis="x",
    bottom=False,
)
# plt.plot(np.linspace(300, 450), np.linspace(300, 450), "k--")
ax.set_ylabel("$p$CO$_2$ ($\mu$atm)")
ax2.set_ylabel("Water Temperature ($^\circ$C)")
plt.xlabel("Measured Sea Surface $p$CO$_2$ ($\mu$atm)")
ax.tick_params(axis="x", labelrotation=45)
plt.savefig("./Plots/mcq_short_timeseries.png")


# %% Mooring Stats
mcq_daily["pred_pCO2"] = np.nan

for i in mcq_daily.index:
    if math.isnan(mcq_daily.loc[i, "TC"]):
        mcq_daily.loc[i, "TC"] = mcq.loc[i, "Temperature.1"]
    this_time = mcq_daily.loc[i, "Time"]
    pred = (
        ds["pco"]
        .isel(
            x=np.argmin(np.abs(ds.x.values + 91.84)),
            y=np.argmin(np.abs(ds.y.values - 46.81)),
        )
        .sel(time=this_time, method="nearest")
        .values
    )
    mcq_daily.loc[i, "pred_pCO2"] = pred

mcq_daily["model_error"] = mcq_daily.pCO2 - mcq_daily.pred_pCO2

early_mcq_error_mean = mcq_daily[0:98].model_error.mean()
late_mcq_error_mean = mcq_daily[98:].model_error.mean()

mcq_error_mean = np.nanmean(np.abs(mcq_daily.model_error))
mcq_error_std = mcq_daily.model_error.std()
# %% Mean conditions

df_mean = pd.DataFrame(
    {
        "time": ds.time.values,
        "pco": ds["pco"].mean(dim=["x", "y"]).values,
        "atmpco": ds["atmpco"].values,
        "temp": ds["temp"].mean(dim=["x", "y"]).values.astype("float"),
        "light": ds["light"].mean(dim=["x", "y"]).values,
        "primprod": ds["primprod"].mean(dim=["x", "y"]).values,
        "airsp": ds["airsp"].mean(dim=["x", "y"]).values,
        "flux": ds["flux_mol_m2_day"].mean(dim=["x", "y"]).values,
        "doy": ds["doy"].values,
    }
)

# df_mean["flux"] = flux_calc(
#    df_mean.pco, df_mean.atmpco, df_mean.airsp, df_mean.temp, 0.045
# )

BH_grouped = pd.read_csv("BH_post_model.csv")
BH_grouped = BH_grouped.loc[BH_grouped.sal < 0.04588465413037274]

# %% Plot means
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.pco,
    c="k",
    label="Model Sea Surface $p$CO$_2$",
    ax=ax,
)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.atmpco,
    c="b",
    label="WLEF Atmospheric $p$CO$_2$",
    ax=ax,
)
ax.set_ylabel("$p$CO$_2$ ($\mu$atm)")
ax.set_xlabel("Julian Day")
plt.tight_layout()
plt.savefig("./Plots/doy_series.png")

# %% Many timeseries
fig, axs = plt.subplots(6, ncols=1, figsize=(12, 10), dpi=300)
# axs[0].set_title("Mean Modeled Conditions")
axs[0].plot(df_mean.time, df_mean.pco, label="Modeled $p$CO$_{2}$")
axs[0].plot(
    df_mean.time, df_mean.atmpco, c="k", label="Atmospheric $p$CO$_{2}$"
)
# BH_plot = axs[0].scatter(
#    pd.to_datetime(BH_grouped.Date, format="ISO8601"),
#    BH_grouped.pCO2,
#    label="BH",
#    c="r",
#    s=5,
#    alpha=0.05,
# )
BH_plot2 = sns.lineplot(
    x=pd.to_datetime(BH_grouped.Date, format="ISO8601"),
    y=BH_grouped.pCO2,
    ax=axs[0],
    errorbar=("se", 1),
    estimator="mean",
    err_style="bars",
    marker="o",
    ls="none",
    c="r",
    label="Underway $p$CO$_{2}$",
)
axs[0].annotate("a", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[0].legend(ncol=3)
axs[0].set_ylim([200, 650])
axs[0].set_xlim([17895, 19724])
axs[0].set_ylabel("$p$CO$_2$\n($\mu atm$)")
axs[0].set(xlabel=None)
axs[0].tick_params(
    axis="x",
    top=False,
    labeltop=False,
    bottom=True,
    labelbottom=False,
)
gs = GridShader(axs[0], facecolor="lightgrey", first=False, alpha=0.7)

axs[1].annotate("b", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[1].plot(df_mean.time, df_mean.temp, c="red")
axs[1].set_ylabel("SST \n($^{\circ}$C)")
axs[1].set_ylim([0, 20])
axs[1].set_xlim([17895, 19724])
axs[1].set(xlabel=None)
axs[1].tick_params(
    axis="x",
    top=False,
    labeltop=False,
    bottom=True,
    labelbottom=False,
)
gs = GridShader(axs[1], facecolor="lightgrey", first=False, alpha=0.7)

axs[2].annotate("c", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[2].plot(df_mean.time, df_mean.light, c="orange")
axs[2].set_ylabel("PAR \n(W m$^{-2}$)")
axs[2].set_ylim([0, 400])
axs[2].set_xlim([17895, 19724])
axs[2].set(xlabel=None)
axs[2].tick_params(
    axis="x",
    top=False,
    labeltop=False,
    bottom=True,
    labelbottom=False,
)
gs = GridShader(axs[2], facecolor="lightgrey", first=False, alpha=0.7)

axs[3].annotate("d", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[3].plot(df_mean.time, df_mean.primprod, c="g")
axs[3].set_ylabel("Prim. Prod. \n(mg C m$^{-2}$ m$^{-1}$)")
axs[3].set_ylim([0, 4])
axs[3].set_xlim([17895, 19724])
axs[3].set(xlabel=None)
axs[3].tick_params(
    axis="x",
    top=False,
    labeltop=False,
    bottom=True,
    labelbottom=False,
)
gs = GridShader(axs[3], facecolor="lightgrey", first=False, alpha=0.7)

axs[4].annotate("e", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[4].plot(df_mean.time, df_mean.airsp, c="purple")
axs[4].set_ylabel("U$_{10}$ \n(m s$^{-1}$)")
axs[4].set_ylim([0, 20])
axs[4].set_xlim([17895, 19724])
axs[4].set(xlabel=None)
axs[4].tick_params(
    axis="x",
    top=False,
    labeltop=False,
    bottom=True,
    labelbottom=False,
)
gs = GridShader(axs[4], facecolor="lightgrey", first=False, alpha=0.7)

# axs[5].plot(df_mean.time, df_mean.flux * 1000, c="grey")
axs[5].annotate("f", xy=(0.01, 0.75), xycoords="axes fraction", fontsize=25)
axs[5].set_ylabel("CO$_2$ Flux \n(mmol C m$^{-2}$ d$^{-1}$)")
axs[5].fill_between(
    df_mean.time, (df_mean.flux.where(df_mean.flux > 0) * 1000), color="red"
)
axs[5].fill_between(
    df_mean.time, (df_mean.flux.where(df_mean.flux < 0) * 1000), color="green"
)
axs[5].set_ylim([-25, 25])
axs[5].set_xlim([17895, 19724])
gs = GridShader(axs[5], facecolor="lightgrey", first=False, alpha=0.7)


plt.tight_layout()

plt.savefig("./Plots/many_timeseries.png")

# %% Prod vs Irrad

fig, ax = plt.subplots()
sns.scatterplot(
    x=df_mean.primprod,
    y=df_mean.temp,
    hue=df_mean.light,
    edgecolor=None,
    palette="viridis",
)
# ax.set_ylabel("Mean Surface Irradiation (W m$^{-2}$)")
# ax.set_xlabel("Mean Primary Production (mg C m$^{-2} d$^{-1})")
sns.despine()
plt.tight_layout()

plt.savefig("./Plots/prod_vs_irrad_vs_temp.png")

# %% GLENDA

g = pd.read_csv("Data/glnpo.csv", parse_dates=["SAMPLING_DATE"])
g = g.loc[g.QC_TYPE == "routine field sample"].loc[g.SAMPLE_DEPTH_M < 15]

glenda = pd.DataFrame(
    {
        "Row": [],
        "YEAR": [],
        "MONTH": [],
        "SEASON": [],
        "LAKE": [],
        "CRUISE_ID": [],
        "VISIT_ID": [],
        "STATION_ID": [],
        "STN_DEPTH_M": [],
        "LATITUDE": [],
        "LONGITUDE": [],
        "SAMPLING_DATE": [],
        "TIME_ZONE": [],
        "SAMPLE_DEPTH_M": [],
        "DEPTH_CODE": [],
        "MEDIUM": [],
        "SAMPLE_TYPE": [],
        "QC_TYPE": [],
        "SAMPLE_ID": [],
        "ANALYTE": [],
        "VALUE": [],
        "UNITS": [],
        "FRACTION": [],
        "METHOD": [],
        "RESULT_REMARK": [],
    }
)
g = g.reset_index(drop=True)
# Reshape I
for i in g.index:
    l = g.iloc[i, 0:19].to_frame().transpose()  # metadata
    e = g.iloc[i, 20:61]
    for j in range(6):
        n = e.iloc[j * 7 : j * 7 + 6].to_frame().transpose()
        a = n.set_axis(
            [
                "ANALYTE",
                "VALUE",
                "UNITS",
                "FRACTION",
                "METHOD",
                "RESULT_REMARK",
            ],
            axis=1,
        )
        d = pd.concat([l, a], axis=1)
        glenda = pd.concat([glenda, d])

glenda = glenda.loc[glenda.VALUE != "No result recorded."]
glenda = glenda.loc[glenda.VALUE != "No result reported."]
glenda = glenda.loc[glenda.VALUE != "no result reported"]
glenda = glenda.loc[glenda.VALUE != "NRR"]
glenda = glenda.loc[glenda.VALUE != "INV 259"]
glenda = glenda.loc[glenda.VALUE != "INV"]
glenda = glenda.loc[glenda.VALUE != "*"]
glenda = glenda.loc[glenda.VALUE != "FAC"]
glenda = glenda.reset_index(drop=True)

for i in glenda.index:
    if str(glenda.loc[i, "ANALYTE"]) == "nan":
        glenda = glenda.drop(i, axis=0)
    elif len(str(glenda.loc[i, "STATION_ID"])) > 5:
        glenda = glenda.drop(i, axis=0)

glenda.VALUE = pd.to_numeric(glenda.VALUE)

# Pivoting

glenda_pivoted = pd.pivot_table(
    glenda,
    index=[
        "YEAR",
        "MONTH",
        "SEASON",
        "SAMPLING_DATE",
        "VISIT_ID",
        "STATION_ID",
        "MEDIUM",
        "SAMPLE_DEPTH_M",
        "LATITUDE",
        "LONGITUDE",
    ],
    columns="ANALYTE",
    values="VALUE",
    aggfunc=np.mean,
)


glenda_pivoted = glenda_pivoted.set_axis(
    ["ta", "ca", "con", "hard", "tc", "ph"], axis=1
)
glenda_pivoted.ta = glenda_pivoted.ta / 1000 / 100.0869 * 1000000 * 2
glenda_pivoted = glenda_pivoted.reset_index()
glenda_tidy = glenda_pivoted.reset_index(drop=True)

glenda_tidy["sal"] = np.nan
glenda_tidy["decimalYEAR"] = np.nan
# glenda_tidy['Tsec'] = (glenda_tidy.SAMPLING_DATE -
#                       dt.datetime(1992, 1, 1)).dt.total_seconds()/60/60/24

for i in glenda_tidy.index:
    glenda_tidy.loc[i, "sal"] = 0.05
    if "August" in glenda_tidy.loc[i, "MONTH"]:
        glenda_tidy.loc[i, "decimalYEAR"] = glenda_tidy.loc[i, "YEAR"] + 8 / 12
    elif "September" in glenda_tidy.loc[i, "MONTH"]:
        glenda_tidy.loc[i, "decimalYEAR"] = glenda_tidy.loc[i, "YEAR"] + 9 / 12
    elif "April" in glenda_tidy.loc[i, "MONTH"]:
        glenda_tidy.loc[i, "decimalYEAR"] = glenda_tidy.loc[i, "YEAR"] + 4 / 12
    elif "May" in glenda_tidy.loc[i, "MONTH"]:
        glenda_tidy.loc[i, "decimalYEAR"] = glenda_tidy.loc[i, "YEAR"] + 5 / 12
#
glenda_yearly = glenda_tidy.copy()
glenda_yearly.YEAR = glenda_yearly.YEAR.astype("str")
results = pyco2.sys(
    par1=glenda_tidy.ta,
    par2=glenda_tidy.ph + 0.137,
    par1_type=1,
    par2_type=3,
    salinity=0.05,
    temperature=25,
    temperature_out=glenda_tidy.tc,
    total_silicate=200,
    total_phosphate=0,
    total_ammonia=0,
    total_borate=0,
    total_fluoride=0,
    total_sulfate=4 * 1000 / 96.06,
    total_calcium=glenda_tidy.ca * 1000 / 40.078,
    opt_pH_scale=4,
    opt_k_carbonic=15,
    uncertainty_into=["pCO2_out"],
    uncertainty_from={"par1": 49.96, "par2": 0.2},
)
glenda_tidy["pCO2"] = results["pCO2_out"]
glenda_tidy["u_pCO2"] = results["u_pCO2_out"]
glenda_tidy["pred_pCO2"] = np.nan

for i in glenda_tidy.index:
    this_lon = glenda_tidy.loc[i, "LONGITUDE"]
    this_lat = glenda_tidy.loc[i, "LATITUDE"]
    this_time = glenda_tidy.loc[i, "SAMPLING_DATE"]
    pred = (
        ds["pco"]
        .isel(
            x=np.argmin(np.abs(ds.x.values - this_lon)),
            y=np.argmin(np.abs(ds.y.values - this_lat)),
        )
        .sel(time=this_time, method="nearest")
        .values
    )
    glenda_tidy.loc[i, "pred_pCO2"] = pred
glenda_tidy["delta_pCO2"] = glenda_tidy.pCO2 - glenda_tidy.pred_pCO2
glenda_tidy = glenda_tidy.dropna(subset="delta_pCO2")
# %% GLENDA plot
fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
sns.scatterplot(
    x=glenda_tidy.pCO2,
    y=glenda_tidy.pred_pCO2,
    hue=glenda_tidy.YEAR,
    style=glenda_tidy.SEASON,
    edgecolor=None,
    s=30,
    alpha=0.5,
    ax=ax,
    palette="tab10",
)
sns.kdeplot(
    x=glenda_tidy.pCO2,
    y=glenda_tidy.pred_pCO2,
)
abline(1, 0, c="k")
ax.set_ylim(250, 500)
ax.set_xlim(0, 1500)
ax.set_ylabel("Model-Predicted $p$CO$_2$ ($\mu$atm)")
ax.set_xlabel("GLENPO-Calculated Sea Surface $p$CO$_2$ ($\mu$atm)")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("./Plots/glenda_compare.png")
# %% GLENDA station compare
fig, ax = plt.subplots(figsize=(4, 6), dpi=150)
sns.barplot(glenda_tidy, y="STATION_ID", x="delta_pCO2")
sns.scatterplot(
    glenda_tidy,
    y="STATION_ID",
    x="delta_pCO2",
    hue="YEAR",
    style="SEASON",
    palette="tab10",
)

# %% GLENDA map

fig = plt.figure(figsize=(12, 8), dpi=250)
ax = fig.add_subplot(
    projection=ccrs.AlbersEqualArea(
        central_longitude=-88, central_latitude=47
    ),
)

ax.set_extent(lims, crs=ccrs.PlateCarree())

# ax.add_feature(cfeature.LAND, facecolor="dimgrey", zorder=1)
ax.add_feature(cfeature.LAKES, zorder=2, facecolor="white", edgecolor="black")
# ax.add_feature(
#    cfeature.RIVERS, zorder=2, facecolor="dimgrey", edgecolor="lightblue"
# )
# ax.add_feature(cfeature.STATES, ls="-.", edgecolor="black", alpha=0.5)
# ax.add_feature(
#    cfeature.NaturalEarthFeature(
#        "physical", "rivers_north_america", "10m"
#    ),
#    facecolor="dimgrey",
#    edgecolor="lightblue",
# )
sns.scatterplot(
    glenda_tidy,
    y="LATITUDE",
    x="LONGITUDE",
    size="delta_pCO2",
    ax=ax,
    transform=ccrs.PlateCarree(),
)
for i in glenda_tidy.index:
    ax.annotate(
        text=glenda_tidy.loc[i, "STATION_ID"],
        xy=(glenda_tidy.loc[i, "LONGITUDE"], glenda_tidy.loc[i, "LATITUDE"]),
        transform=ccrs.PlateCarree(),
    )
ax.set_axis_off()
gl = ax.gridlines(
    draw_labels=True,
    ls=":",
    alpha=0.5,
    xlocs=np.arange(-92, -82, 2),
    ylocs=np.arange(46, 50, 1),
)
gl.right_labels = False
gl.top_labels = False

fig.tight_layout()
# %% Import bottle samples

bottles = pd.read_csv(
    "Inorganic Carbon Master Logsheet - Logsheet_for_plotting.csv",
    dtype={"CTDTEMP": float},
)
bottles["S"] = gsw.conversions.SP_from_C(
    bottles["CTDCON"] / 1000, bottles["CTDTEMP"], bottles["Depth"]
)
for i in bottles.index:
    if math.isnan(bottles.loc[i, "S"]):
        bottles.loc[i, "S"] = bottles.loc[i, "CTDSAL_PSS78"]
    if math.isnan(bottles.loc[i, "TA"]):
        bottles.loc[i, "TA_ass"] = 842.7

sns.lmplot(data=bottles, x="S", y="TA")
bottles = bottles[bottles.Depth < 4]


# %% Match bottles to pred

for i in bottles.index:
    this_lon = float(bottles.loc[i, "Longitude"])
    this_lat = float(bottles.loc[i, "Latitude"])
    this_time = bottles.loc[i, "Sample_Date_UTC"]
    nearest_x = np.argmin(np.abs(ds.x.values - this_lon))
    nearest_y = np.argmin(np.abs(ds.y.values - this_lat))
    try:
        pred = (
            ds["pco"]
            .isel(
                x=[nearest_x - 2, nearest_x, nearest_x + 2],
                y=[nearest_y - 2, nearest_y, nearest_y + 2],
            )
            .sel(time=this_time, method="nearest")
            .values
        )
        bottles.loc[i, "pred_pco2"] = np.nanmean(pred)
        pred_temp = (
            ds["temp"]
            .isel(
                x=[nearest_x - 2, nearest_x, nearest_x + 2],
                y=[nearest_y - 2, nearest_y, nearest_y + 2],
            )
            .sel(time=this_time, method="nearest")
            .values
        )
        bottles.loc[i, "pred_temp"] = np.nanmean(pred_temp)
    except Exception as error:
        # handle the exception
        print(
            "An exception occurred:", error
        )  # An exception occurred: division by zero

# %% bottles CO2SYS
kwargs = {
    "pressure_out": 0,
    "opt_k_carbonic": 15,
    "opt_pH_scale": 3,
    "pressure": bottles.Depth,
    "salinity": bottles.S,
    "total_silicate": 200,
    "total_phosphate": 0,
    "total_borate": 0,
    "total_calcium": 14 * 1000 / 40.078,
    "total_fluoride": 0,
    "total_sulfate": 4 * 1000 / 96.06,
    "uncertainty_into": ["pCO2", "pCO2_out"],
}

results = pyco2.sys(
    par1=bottles.DIC,
    par2=bottles.TA,
    par1_type=2,
    par2_type=1,
    temperature=bottles.CTDTEMP,
    temperature_out=bottles.CTDTEMP,
    uncertainty_from={"par1": 8, "par2": 5.3, "temperature": 1},
    **kwargs
)
bottles["pco2_dicta"] = results["pCO2_out"]
bottles["u_pco2_dicta"] = results["u_pCO2_out"]
bottles["deltapco2_dicta"] = bottles["pco2_dicta"] - bottles["pred_pco2"]


results = pyco2.sys(
    par1=bottles.DIC,
    par2=bottles.pH_measured,
    par1_type=2,
    par2_type=3,
    temperature=bottles.TEMP_pH,
    temperature_out=bottles.CTDTEMP,
    uncertainty_from={"par1": 8, "par2": 0.01, "temperature": 1},
    **kwargs
)
bottles["pco2_dicph"] = results["pCO2_out"]
bottles["u_pco2_dicph"] = results["u_pCO2_out"]
bottles["deltapco2_dicph"] = bottles["pco2_dicph"] - bottles["pred_pco2"]


results = pyco2.sys(
    par1=bottles.TA,
    par2=bottles.pH_measured,
    par1_type=1,
    par2_type=3,
    temperature=bottles.TEMP_pH,
    temperature_out=bottles.CTDTEMP,
    uncertainty_from={"par1": 5.3, "par2": 0.01, "temperature": 0.1},
    **kwargs
)
bottles["pco2_taph"] = results["pCO2_out"]
bottles["u_pco2_taph"] = results["u_pCO2_out"]
bottles["deltapco2_taph"] = bottles["pco2_taph"] - bottles["pred_pco2"]


results = pyco2.sys(
    par1=bottles.TA_ass,
    par2=bottles.pH_measured,
    par1_type=1,
    par2_type=3,
    temperature=bottles.TEMP_pH,
    temperature_out=bottles.CTDTEMP,
    uncertainty_from={"par1": 20.5, "par2": 0.01, "temperature": 0.1},
    **kwargs
)
bottles["pco2_asstaph"] = results["pCO2_out"]
bottles["u_pco2_asstaph"] = results["u_pCO2_out"]
bottles["deltapco2_asstaph"] = bottles["pco2_asstaph"] - bottles["pred_pco2"]

# %% Plot bottle errors
flag_limit = 5

fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

ax.errorbar(
    x=bottles.pred_pco2,
    y=bottles.pco2_dicph,
    yerr=bottles.u_pco2_dicph,
    xerr=26,
    fmt="g.",
    label="DIC-pH",
)

"""
ax.errorbar(
    x=bottles.pred_pco2,
    y=bottles.pco2_dicta,
    yerr=bottles.u_pco2_dicta,
    xerr = 26,
    fmt="k.",
    label="A$_T$-DIC",
)
"""

ax.errorbar(
    x=bottles.pred_pco2,
    y=bottles.pco2_taph,
    yerr=bottles.u_pco2_taph,
    xerr=26,
    fmt="b.",
    label="A$_T$-pH",
)
"""
ax.errorbar(
    x=bottles.pred_pco2,
    y=bottles.pco2_asstaph,
    yerr=bottles.u_pco2_asstaph,
    xerr = 26,
    fmt="r.",
    label="A$_T assumed$-$p$H",
)
"""
ax.set_ylim(250, 550)
ax.set_xlim(250, 550)
abline(1, 0, c="k", label="1:1 Line")
ax.set_ylabel("Bottle Calculated $p$CO$_2$ ($\mu$atm)")
ax.set_xlabel("Model Predicted $p$CO$_2$ ($\mu$atm)")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# %% Both bottles
fig, ax = plt.subplots(figsize=(4, 4), dpi=250)

# sns.residplot(data = bottles, y = 'pred_pco2', x = 'pco2_dicta', ax = ax, label = 'Authors: DIC-TA')

sns.scatterplot(
    x=glenda_tidy.pred_pCO2,
    y=glenda_tidy.delta_pCO2,
    label="GLNPO: A$_T$-pH (n=202)",
    color="r",
    s=12,
    edgecolor=None,
    ax=ax,
)
sns.scatterplot(
    data=bottles,
    x="pred_pco2",
    y="deltapco2_taph",
    ax=ax,
    marker="s",
    label="this work: A$_T$-pH  (n=6)",
    color="b",
    edgecolor=None,
)
sns.scatterplot(
    data=bottles,
    x="pred_pco2",
    y="deltapco2_dicph",
    ax=ax,
    marker="^",
    label="this work: DIC-pH (n=11)",
    color="g",
    edgecolor=None,
)

ax.set_ylim(-250, 1200)
ax.set_xlim(275, 500)
ax.set_ylabel("$\Delta$$p$CO$_2$ ($\mu$atm)")
ax.set_xlabel("Model Predicted $p$CO$_2$ ($\mu$atm)")
abline(0, 0, color="k", ls=":")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("./Plots/all_bottles_compare.png")
# %% Bottle Stats
bottles_used = bottles.dropna(how="all", subset=["pco2_dicph", "pco2_taph"])
bottle_err = np.vstack(
    (
        np.array(bottles[["pco2_dicph", "pred_pco2"]].dropna()),
        np.array(bottles[["pco2_taph", "pred_pco2"]].dropna()),
        # np.array(bottles[["pco2_asstaph", "pred_pco2"]].dropna()),
    )
)

Y = bottle_err[:, 0]
X = bottle_err[:, 1]
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
results.summary()
bottle_rmse = sm.tools.eval_measures.rmse(bottle_err[:, 0], bottle_err[:, 1])
# %% Seasonal Decomp
# Can't see a trend in mean pCO2 for the life of me
res = MSTL(df_mean.pco, periods=(365, 180, 90, 28), iterate=3).fit()
fig = res.plot()
fig.set_dpi(200)
fig.set_size_inches(5, 8)
fig.tight_layout()
# But it's easy to find in the atmospheric signal.
res = MSTL(df_mean.atmpco, periods=(365, 180, 200), iterate=3).fit()
fig = res.plot()
fig.set_dpi(200)
fig.set_size_inches(5, 8)
fig.tight_layout()

# %% T vs. B Climatology

dlnpCO2dT = 0.0360604

df_mean["pCO2_T"] = np.nan
df_mean["pCO2_B"] = np.nan
mean_pco2 = df_mean.pco.mean()
mean_temp = df_mean.temp.mean()
for i in df_mean.index:
    df_mean.loc[i, "pCO2_T"] = mean_pco2 * np.exp(
        dlnpCO2dT * (df_mean.temp[i] - mean_temp)
    )
    df_mean.loc[i, "pCO2_B"] = df_mean.pco[i] * np.exp(
        dlnpCO2dT * (mean_temp - df_mean.temp[i])
    )


fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.pCO2_B,
    c="#4daf4a",
    label="Non-Thermal $p$CO$_2$",
    ax=ax,
)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.pco,
    c="#377eb8",
    label="Model $p$CO$_2$",
    ax=ax,
)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.pCO2_T,
    c="#e41a1c",
    label="Thermal $p$CO$_2$",
    ax=ax,
)
sns.lineplot(
    x=df_mean.doy,
    y=df_mean.atmpco,
    c="k",
    ls=":",
    label="WLEF Atmospheric $p$CO$_2$",
    ax=ax,
)
ax.set_ylabel("$p$CO$_2$ ($\mu$atm)")
ax.set_xlabel("Julian Day")
plt.tight_layout()
plt.savefig("./Plots/TvB_series.png")

print("T vs. B climatology plotted.")
