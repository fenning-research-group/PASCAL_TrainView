from frgpascal.analysis.processing import load_all, compress_jv, get_worklist_times
from scipy import stats
from natsort import natsorted
from natsort import index_natsorted
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colorbar
import seaborn as sns
import warnings
from tqdm import tqdm
from matplotlib import style
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import os
import json
import pickle as pkl
import time

# plotting settings
# %config InlineBackend.figure_format = 'retina'
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["axes.linewidth"] = 1.75  # set the value globally

##


def get_additional_params(paramdf):
    data = {}
    # data['storage_tray'] = []
    data["sample_number"] = []
    data["substrate"] = []

    # data['spincoat1_drop1_time'] = []
    # data['spincoat1_drop1_rate'] = []
    # data['spincoat1_drop1_height'] = []
    # data['spincoat1_drop1_volume'] = []

    for n in range(len(paramdf)):
        sample_number = paramdf["name"][n]
        substrate = paramdf["substrate"][n]
        # storage_tray = paramdf['storage_tray'][n]

        # spincoat1_drop1_time = paramdf['spincoat1_drop1_time'][n]
        # spincoat1_drop1_rate = paramdf['spincoat1_drop1_rate'][n]
        # spincoat1_drop1_height = paramdf['spincoat1_drop1_height'][n]
        # spincoat1_drop1_volume = paramdf['spincoat1_drop1_volume'][n]

        data["sample_number"].append(sample_number)
        data["substrate"].append(substrate)

        # data['spincoat1_drop1_time'].append(spincoat1_drop1_time)
        # data['spincoat1_drop1_rate'].append(spincoat1_drop1_rate)
        # data['spincoat1_drop1_height'].append(spincoat1_drop1_height)
        # data['spincoat1_drop1_volume'].append(spincoat1_drop1_volume)

    return data


##


def load_all_sorted(chardir):
    metricdf, rawdf = load_all(chardir, t_kwargs=dict(wlmin=700, wlmax=900))
    rawdf = rawdf.sort_values(
        by="name", key=lambda x: np.argsort(index_natsorted(rawdf["name"]))
    )
    rawdf = rawdf.reset_index(drop=True)

    metricdf = metricdf.sort_values(
        by="name", key=lambda x: np.argsort(index_natsorted(metricdf["name"]))
    )
    metricdf = metricdf.reset_index(drop=True)
    return metricdf, rawdf


##


def adjust_time(timedf):
    timedf_0 = timedf

    time_list = []

    for n in range(len(timedf_0)):
        time_list.append(timedf_0["spincoat0"][n][0])

    for n in range(len(timedf_0)):
        timedf_0["spincoat0"][n] = time_list[n]

    return timedf_0


##


def rename_duplicate_cols(df):
    df = df
    cols = pd.Series(df.columns)

    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            dup + "." + str(i) if i != 0 else dup for i in range(sum(cols == dup))
        ]

    # rename the columns with the cols list.
    df.columns = cols
    return df


##


def correlation_plot(metricdf, x_col=str, y_col=str, batch=str, save=True):
    save = save
    batch = batch
    metricdf = metricdf
    fig, ax = plt.subplots()
    x = metricdf[x_col].astype(float)
    y = metricdf[y_col].astype(float)
    sns.scatterplot(x=x, y=y, ax=ax, color="black", alpha=1, legend=None)
    sns.kdeplot(
        x=x, y=y, cmap="Greys_r", shade=True, bw_method="scott", ax=ax, alpha=0.2
    )
    res = stats.linregress(x, y)
    rsq = res.rvalue**2
    ax.plot(
        x, res.intercept + res.slope * x, "r"
    )  # , label=f'R$^2$:{rsq:.2f}', color='springgreen')
    plt.text(
        0.01,
        0.95,
        s=(f"R$^2$:{rsq:.2f}"),
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        color="red",
        weight="bold",
    )
    plt.ylabel(y.name, size=15)
    plt.xlabel(x.name, size=15)

    TodaysDate = time.strftime("%Y%m%d")
    if save == True:
        plt.savefig(
            f"{TodaysDate}_{batch}_{x_col}_{y_col}.png", dpi=300, bbox_inches="tight"
        )


##


def correlation_matrix(metricdf, method="pearson", batch=str, save=True):
    save = save
    metricdf = metricdf
    method = method
    batch = batch
    columns = [
        "pce_f",
        "pce_r",
        "ff_f",
        "ff_r",
        "voc_f",
        "voc_r",
        "jsc_f",
        "jsc_r",
        "pl_intensity_0",
        "pl_peakev_0",
        "pl_fwhm_0",
        "t_bandgap_0",
        "spincoat0",
    ]
    d = metricdf[columns]
    d = d.apply(pd.to_numeric, errors="coerce")

    # Compute the correlation matrix
    corr = d.corr(
        method=method
        # method='kendall'
    )

    # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    mask = np.eye(corr.shape[0])

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        # vmax=.7,
        # vmin=-0.7,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Pearson Correlation"},
    )
    TodaysDate = time.strftime("%Y%m%d")
    if save == True:
        plt.savefig(
            f"{TodaysDate}_{batch}_correlation_matrix.png", dpi=300, bbox_inches="tight"
        )


##


def factor(x):
    factor_list = []
    for i in range(1, x + 1):
        if x % i == 0:
            factor_list.append(i)
    return factor_list


def shape_plot(rawdf, x_aspect=8):
    count = len(rawdf)

    while (count % x_aspect) != 0:
        count += 1

    first_list = []
    second_list = []
    add_list = []
    factor_list = factor(count)
    for n in range(len(factor_list)):
        first = factor_list[n]
        first_list.append(first)
        for m in range(len(factor_list)):
            second = factor_list[m]
            if first * second == count:
                second_list.append(second)
                add_list.append(first + second)

    first_list = np.array(first_list)
    second_list = np.array(second_list)
    add_list = np.array(add_list)
    return second_list[np.argmin(add_list)], first_list[np.argmin(add_list)]


##


def plot_bf(rawdf, batch=str, save=True):
    save = save
    batch = batch
    blank = rawdf["bf_0"][0] * 0
    horiz, vert = shape_plot(rawdf, 8)

    embiggen = 2
    item = 0

    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=False,
    )

    for k in range(horiz):
        for n in range(vert):
            try:
                name = rawdf["name"][item]
                pl_intensity_0 = rawdf["pl_intensity_0"][item]
                pl_peakev_0 = rawdf["pl_peakev_0"][item]
                ax[n, k].imshow(rawdf["bf_0"][item])
            except:
                ax[n, k].imshow(blank)
                name = "eblank"
                pl_intensity_0 = 0
                pl_peakev_0 = 0

            try:
                pce_r = rawdf["pce_f"][item]
                ff_r = rawdf["ff_f"][item]
            except:

                pce_r = 0
                ff_r = 0

            plt.text(
                0.01,
                1,
                s=(name.split("e")[1]),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Red",
                weight="bold",
                backgroundcolor="White",
            )
            plt.text(
                0.01,
                0.35,
                s=(f"PCEf: {pce_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )
            plt.text(
                0.01,
                0.45,
                s=(f"FFf: {ff_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )

            plt.text(
                0.01,
                0.15,
                s=(f"PL-Cts: {pl_intensity_0:.1f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Yellow",
                weight="bold",
            )
            plt.text(
                0.01,
                0,
                s=(f"PL-eV: {pl_peakev_0:.2f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Black",
                weight="bold",
            )

            item += 1
            ax[n, k].set_xticks([])
            ax[n, k].set_yticks([])
            ax[n, k].axis("off")
    fig.subplots_adjust(wspace=0, hspace=-0.3)
    TodaysDate = time.strftime("%Y%m%d")
    if save == True:
        plt.savefig(f"{TodaysDate}_{batch}_camera_bf.png", dpi=300, bbox_inches="tight")


def plot_df(rawdf, batch=str, save=True):
    save = save
    batch = batch
    blank = rawdf["bf_0"][0] * 0
    horiz, vert = shape_plot(rawdf, 8)

    embiggen = 2
    item = 0

    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=False,
    )

    for k in range(horiz):
        for n in range(vert):
            try:
                name = rawdf["name"][item]
                pl_intensity_0 = rawdf["pl_intensity_0"][item]
                pl_peakev_0 = rawdf["pl_peakev_0"][item]
                ax[n, k].imshow(rawdf["df_0"][item][:, :, 0], cmap="gray")
            except:
                ax[n, k].imshow(blank)
                name = "eblank"
                pl_intensity_0 = 0
                pl_peakev_0 = 0

            try:
                pce_r = rawdf["pce_f"][item]
                ff_r = rawdf["ff_f"][item]
            except:

                pce_r = 0
                ff_r = 0

            plt.text(
                0.01,
                1,
                s=(name.split("e")[1]),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Red",
                weight="bold",
                backgroundcolor="White",
            )
            plt.text(
                0.01,
                0.35,
                s=(f"PCEf: {pce_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )
            plt.text(
                0.01,
                0.45,
                s=(f"FFf: {ff_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )

            plt.text(
                0.01,
                0.15,
                s=(f"PL-Cts: {pl_intensity_0:.1f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Yellow",
                weight="bold",
            )
            plt.text(
                0.01,
                0,
                s=(f"PL-eV: {pl_peakev_0:.2f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Black",
                weight="bold",
            )

            item += 1
            ax[n, k].set_xticks([])
            ax[n, k].set_yticks([])
            ax[n, k].axis("off")
    fig.subplots_adjust(wspace=0, hspace=-0.3)
    TodaysDate = time.strftime("%Y%m%d")
    if save == True:
        plt.savefig(f"{TodaysDate}_{batch}_camera_df.png", dpi=300, bbox_inches="tight")


def plot_pl(rawdf, batch=str, save=True):
    save = save
    batch = batch
    blank = rawdf["bf_0"][0] * 0
    horiz, vert = shape_plot(rawdf, 8)

    embiggen = 2
    item = 0

    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=False,
    )

    for k in range(horiz):
        for n in range(vert):
            try:
                name = rawdf["name"][item]
                pl_intensity_0 = rawdf["pl_intensity_0"][item]
                pl_peakev_0 = rawdf["pl_peakev_0"][item]
                ax[n, k].imshow(rawdf["plimg_0"][item], cmap="viridis")
            except:
                ax[n, k].imshow(blank)
                name = "eblank"
                pl_intensity_0 = 0
                pl_peakev_0 = 0

            try:
                pce_r = rawdf["pce_f"][item]
                ff_r = rawdf["ff_f"][item]
            except:

                pce_r = 0
                ff_r = 0

            plt.text(
                0.01,
                1,
                s=(name.split("e")[1]),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Red",
                weight="bold",
                backgroundcolor="White",
            )
            plt.text(
                0.01,
                0.35,
                s=(f"PCEf: {pce_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )
            plt.text(
                0.01,
                0.45,
                s=(f"FFf: {ff_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="lime",
                weight="bold",
            )

            plt.text(
                0.01,
                0.15,
                s=(f"PL-Cts: {pl_intensity_0:.1f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Yellow",
                weight="bold",
            )
            plt.text(
                0.01,
                0,
                s=(f"PL-eV: {pl_peakev_0:.2f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Black",
                weight="bold",
            )

            item += 1
            ax[n, k].set_xticks([])
            ax[n, k].set_yticks([])
            ax[n, k].axis("off")
    fig.subplots_adjust(wspace=0, hspace=-0.3)
    TodaysDate = time.strftime("%Y%m%d")
    if save == True:
        plt.savefig(f"{TodaysDate}_{batch}_camera_pl.png", dpi=300, bbox_inches="tight")


##


def baseline_analysis(
    batch=str,
    chardir=str,
    paramdir=str,
    logdir=str,
    jvdir=None,
    drop_low_pl=50,
    save=True,
    metricdf=None,
    rawdf=None,
):
    chardir_0 = chardir
    paramdir_0 = paramdir
    logdir_0 = logdir
    jvdir_0 = jvdir
    save = save
    batch = batch

    TodaysDate = time.strftime("%Y%m%d")
    fp = "{}_{}_analysis".format(TodaysDate, batch)
    if not os.path.exists(fp):
        os.mkdir(fp)
    os.chdir(fp)

    if metricdf is None:
        paramdf_0 = pd.read_csv(paramdir_0)
        paramdf_0 = paramdf_0.sort_values(
            by="name", key=lambda x: np.argsort(index_natsorted(paramdf_0["name"]))
        )
        paramdf_0 = paramdf_0.reset_index(drop=True)
        metricdf_0, rawdf_0 = load_all_sorted(chardir_0)
        timedf_0 = adjust_time(get_worklist_times(logdir_0))

        if jvdir_0 != None:
            jvdf_0 = compress_jv(jvdir_0)
            test0 = pd.concat([paramdf_0, metricdf_0], axis=1)
            test1 = pd.concat([timedf_0, jvdf_0], axis=1)
            test2 = pd.concat([test0, test1], axis=1)
            test3 = pd.concat([test2, rawdf_0], axis=1)
        if jvdir_0 == None:
            test0 = pd.concat([paramdf_0, metricdf_0], axis=1)
            test2 = pd.concat([test0, timedf_0], axis=1)
            test3 = pd.concat([test2, rawdf_0], axis=1)

        test2, test3 = rename_duplicate_cols(test2), rename_duplicate_cols(test3)

        test2 = test2[~(test2["pl_intensity_0"] <= drop_low_pl)]

        metricdf, rawdf = (
            test2,
            test3,
        )  # .dropna would make correlation plots work better, but this shows all data
        rawdf = rawdf.reset_index(drop=True)

    # chronoglical plots
    correlation_plot(
        metricdf, x_col="spincoat0", y_col="pl_intensity_0", batch=batch, save=save
    )
    correlation_plot(
        metricdf, x_col="spincoat0", y_col="pl_peakev_0", batch=batch, save=save
    )
    correlation_plot(
        metricdf, x_col="spincoat0", y_col="pl_fwhm_0", batch=batch, save=save
    )

    if metricdf["pce_f"].isnull().values.any() == True:
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="pce_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="pce_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="ff_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="ff_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="voc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="voc_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="jsc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="spincoat0", y_col="jsc_r", batch=batch, save=save
        )

        # compare to PL
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="pce_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="pce_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="ff_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="ff_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="voc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="voc_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="jsc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_intensity_0", y_col="jsc_r", batch=batch, save=save
        )

        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="pce_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="pce_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="ff_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="ff_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="voc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="voc_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="jsc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_peakev_0", y_col="jsc_r", batch=batch, save=save
        )

        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="pce_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="pce_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="ff_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="ff_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="voc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="voc_r", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="jsc_f", batch=batch, save=save
        )
        correlation_plot(
            metricdf, x_col="pl_fwhm_0", y_col="jsc_r", batch=batch, save=save
        )

    correlation_matrix(metricdf, method="pearson", batch=batch, save=save)
    plot_df(rawdf, batch=batch, save=save)
    plot_bf(rawdf, batch=batch, save=save)
    plot_pl(rawdf, batch=batch, save=save)
    os.chdir("..")
    return metricdf, rawdf


##
