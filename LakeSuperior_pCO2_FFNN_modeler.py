#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:05:02 2023

@author: sandb425
"""
import numpy as np  # For numerical fast numerical calculations
import matplotlib.pyplot as plt  # For making plots
import pandas as pd  # Deals with data
import seaborn as sns  # Makes better plots
from sklearn.preprocessing import (  # ML tools
    StandardScaler,
)  # Testing sklearn
from keras.models import Sequential  # ML API
from keras.layers import Dense, Dropout  # , BatchNormalization
from keras import callbacks
from sklearn.model_selection import train_test_split  # model validation
import xarray as xr  # NDarray manipulation
from joblib import Parallel, delayed  # parallelization
import pyseaflux as sf  # flux calc harmonization
import shap

# %% Functions


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


def abline(slope, intercept, **kwargs):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, "-", **kwargs)


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


def parallel_predictor(i, j):  # , model, scaler, da):
    X_da = scaler.transform(
        np.array(
            [
                ds.temp.sel(x=i, y=j).values,
                ds.light.sel(x=i, y=j).values,
                ds.primprod.sel(x=i, y=j).values,
                ds.airsp.sel(x=i, y=j).values,
            ]
        ).T
    )
    if np.isnan(np.min(X_da)):
        print("That's a nan for " + str(i) + " " + str(j))
    else:
        try:
            ds["pco"].loc[dict(x=i, y=j)] = model.predict(X_da).ravel()
            print("Sucessful prediction for " + str(i) + " " + str(j))
        except:
            # da["pco"].loc[dict(x=i, y=j)] = np.nan
            print("Error for " + str(i) + " " + str(j))


def BH_model_construction():
    BH_grouped = pd.read_csv("BH1923_Processed_Rad.csv")
    BH_grouped["latish"] = round(BH_grouped["lat"], 3)
    BH_grouped["lonish"] = round(BH_grouped["lon"], 3)
    BH_grouped["wind"] = BH_grouped["TrueWind-SPEED"] / 1.94384  # kts to m/s
    BH_grouped["production"] = np.nan

    for i in BH_grouped.index:
        BH_grouped.loc[i, "production"] = S10_PP(
            BH_grouped.loc[i, "rad"], BH_grouped.loc[i, "Temp"]
        )

    BH_grouped = BH_grouped.groupby(by=["Date", "latish", "lonish"]).agg(
        number=("pCO2", "count"),
        uagg=("pCO2", "std"),
        pCO2=("pCO2", "mean"),
        pCO2atm=("pCO2atm", "mean"),
        Temp=("Temp", "mean"),
        Time=("Time", "mean"),
        production=("production", "mean"),
        rad=("rad", "mean"),
        wind=("wind", "mean"),
        sal=("sal", "mean"),
    )

    BH_grouped.reset_index(inplace=True)
    BH_grouped["doy"] = BH_grouped.apply(
        lambda x: doy(
            pd.to_datetime(x.Date, format="ISO8601").year,
            pd.to_datetime(x.Date, format="ISO8601").month,
            pd.to_datetime(x.Date, format="ISO8601").day,
        ),
        axis=1,
    )
    # DOY sensitivity analysis
    BH_grouped = BH_grouped.loc[BH_grouped.doy > 121].loc[BH_grouped.doy < 275]

    BH_grouped.to_csv("BH_post_model.csv")

    # Prepare
    prediction_parameters = [
        "Temp",
        "rad",
        "production",
        "wind",
    ]

    # drop missing cases
    BH_grouped = BH_grouped.dropna(subset=["pCO2"])
    BH_grouped = BH_grouped.dropna(subset=prediction_parameters)

    # Create our label y:
    y = BH_grouped[["pCO2"]]

    X_numerical = BH_grouped[prediction_parameters]

    # Create all features
    X = X_numerical
    feature_names = X.columns

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Data standardization
    scaler = StandardScaler().fit(X_train[feature_names].values)

    X_train[feature_names] = scaler.transform(X_train[feature_names])
    X_test[feature_names] = scaler.transform(X_test[feature_names])
    # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    # X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

    # Make contiguous flattened arrays (for our scikit-learn model)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Regression
    # Create the model
    n_features = np.shape(X_train)[1]
    model = Sequential()
    # model.add(BatchNormalization())

    # Add the first hidden layer
    model.add(Dense(128, input_shape=(n_features,), activation="relu"))
    # model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))
    # Add the second hidden layer
    model.add(Dense(32, activation="relu"))
    # model.add(Dropout(0.1))

    # model.add(Dense(32, activation="relu"))
    # model.add(Dropout(0.1))

    # model.add(Dense(16, activation="relu"))

    # Add the output layer
    model.add(Dense(1, activation="linear"))

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            "mean_squared_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
        ],
    )

    # Fit the model to the data
    callback = callbacks.EarlyStopping(monitor="loss", patience=5)
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        verbose=1,
        callbacks=[callback],
    )

    # Evaluate the model on the test data
    score = model.evaluate(X_test, y_test)
    for name, value in zip(model.metrics_names, score):
        print(name, value)
        if name == "mean_squared_error":
            print("mean_error ", np.sqrt(value))

    print("Test score:", score)

    # Prediction
    y_pred = model.predict(X_test).ravel()

    # Obs vs Pred
    sns.set(rc={"figure.figsize": (4, 4)})
    sns.set_style("ticks")
    sns.despine()
    
    fig = sns.jointplot(
        x=y_test, y=y_pred, s=1, color="k", alpha=0.7, linewidth=0
    )
    fig.ax_marg_y.set_ylim(200, 600)
    fig.ax_marg_x.set_xlim(200, 600)
    plt.ylabel("Predicted Sea Surface $p$CO$_2$ (\u03bcatm)")
    plt.xlabel("Measured Sea Surface $p$CO$_2$ (\u03bcatm)")
    plt.annotate("1:1 line", xy=(535, 525), fontsize=8, fontstyle="italic")
    plt.annotate(
        "$u_{model}$: " + str(round(score[2], 1)) + " \u03bcatm",
        xy=(210, 575),
    )
    abline(1, 0, c="k", alpha=0.2)
    try:
        plt.savefig("./Plots/BH_FFNN_model.png")
    except:
        print("BH model isn't plotting.")

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    sns.residplot(
        x=y_test,
        y=y_pred,
        scatter_kws={"s": 2, "color": "k", "alpha": 0.004, "marker": "o"},
        ax=ax,
    )
    ax.set_ylim(-100, 100)
    ax.set_xlim(200, 600)
    plt.ylabel("$\Delta p$CO$_2$ (\u03bcatm)")
    plt.xlabel("Measured Sea Surface $p$CO$_2$ (\u03bcatm)")

    # SHAP
    X_train_summary = shap.sample(X_train, 1000)  # shap.kmeans(X_train, 50)
    # compute SHAP values
    explainer = shap.Explainer(model.predict, X_train_summary)
    shap_values = explainer(X_train_summary)
    shap.plots.violin(
        shap_values,
        feature_names=["SST", "PAR", "Prim. Prod.", "U$_{10}$"],
        plot_type="layered_violin",
        plot_size=(12, 6),
    )
    shap.plots.bar(shap_values)
    shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1))

    return model, fig, y_pred, scaler


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
    # cm/hr * mol/L/atm * \u03bcatm * atm/10^6 \u03bcatm * 1000 L/m^3 / 100 cm/m * 1000 mmol/mol * 24 hr/day = mmol/m^2/day
    flux = (
        littlek
        * Ko
        * (pco2w - pco2atm[:, None, None])
        / 1e6
        * 1000
        / 100
        * 1000
        * 24
    )
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
        ).fillna(
            1
        )  # because of missing ice values in summer '23
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
    return ds, flux_integrated


# %% Put it all together and apply to fields

if __name__ == "__main__":
    print("Beginning model construction.")
    model, fig, y_pred, scaler = BH_model_construction()
    print("Model developed.")
    # put all of the new_fields into one monster dataset
    print("Opening all input field files.")
    ds = xr.open_dataset("./bhuw_input.nc")  # new regridded dataset
    # run prediction in parallel over space
    print("Starting prediction for input fields using multiprocessing.")
    cores = 8  # os.cpu_count()
    print(str(cores) + " cores in use.")
    
    Parallel(n_jobs=cores, prefer="threads")(
        delayed(parallel_predictor)(i, j)  # , model, scaler, ds)
        for i in ds.x.values
        for j in ds.y.values
    )

   
    print("Finished prediction.")

    ds, total_annual_flux = pyflux_calc(ds, ice_corr=True)  # GgC/year
    print("Total Annual Flux: " + str(total_annual_flux) + "TgC/yr")

    print("Completed prediction. Saving output as regression_output.nc")
    ds.to_netcdf("regression_output.nc", mode="w")
    print("Producing and saving plot of pCO2 at time = 0.")
    try:
        t0plot = ds["pco"].isel(time=0).plot()
        t0plot.savefig("t0plot.png")

    except:
        print("Plotting is broken.")
    print("Script completed.")
