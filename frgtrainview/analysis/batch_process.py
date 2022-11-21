from frgpascal.analysis.processing import load_all, compress_jv, get_worklist_times
from frgtrainview.analysis import crop
import scipy
from scipy.interpolate import griddata
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

    x_name = x_col
    y_name = y_col
    x = metricdf[x_name]
    y = metricdf[y_name]
    xy = pd.concat([x, y], axis=1)
    xy = xy.dropna()
    x = xy[x_name].astype(float)
    y = xy[y_name].astype(float)
    x_len = len(x)

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
        s=(f" R$^2$:{rsq:.2f}\n N:{x_len}"),
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
    plt.close()


##


def correlation_matrix_jv(metricdf, method="pearson", batch=str, save=True):
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
        "rsh_f",
        "rsh_r",
        "rs_f",
        "rs_r",
        "rch_f",
        "rch_r",
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
    plt.close()


def correlation_matrix(metricdf, method="pearson", batch=str, save=True):
    save = save
    metricdf = metricdf
    method = method
    batch = batch
    columns = [
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
    plt.close()


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
    plt.close()


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
    plt.close()


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
                ax[n, k].imshow(
                    crop.crop_pl(np.uint8(rawdf["plimg_0"][item] * 255))[0],
                    cmap="viridis",
                )
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
                color="Black",
                weight="bold",
            )
            plt.text(
                0.01,
                0.45,
                s=(f"FFf: {ff_r:.2f}%"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Black",
                weight="bold",
            )

            plt.text(
                0.01,
                0.15,
                s=(f"PL-Cts: {pl_intensity_0:.1f}"),
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[n, k].transAxes,
                color="Black",
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
    plt.close()


##


def baseline_analysis(
    batch=str,
    chardir=str,
    paramdir=str,
    logdir=str,
    jvdir=None,
    drop_low_pl=50,
    drop_high_pl=None,
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
    if save == True:
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

        if drop_low_pl != None:
            test2 = test2[~(test2["pl_intensity_0"] <= drop_low_pl)]
            test3 = test3[~(test3["pl_intensity_0"] <= drop_low_pl)]
        if drop_high_pl != None:
            test2 = test2[~(test2["pl_intensity_0"] >= drop_high_pl)]
            test3 = test3[~(test3["pl_intensity_0"] >= drop_high_pl)]
        metricdf, rawdf = (
            test2,
            test3,
        )
        rawdf = rawdf.reset_index(drop=True)

        if jvdir_0 != None:
            metricdf_dropped = metricdf.dropna(subset=["pce_r"])

    # chronoglical plots
    chrono_yvar_list_pl = ["pl_intensity_0", "pl_peakev_0", "pl_fwhm_0"]
    chrono_yvar_list_jv = [
        "pce_f",
        "pce_r",
        "ff_f",
        "ff_r",
        "voc_f",
        "voc_r",
        "jsc_f",
        "jsc_r",
        "rs_f",
        "rs_r",
        "rsh_f",
        "rsh_r",
        "rch_f",
        "rch_r",
        "i_factor_r",
        "i_factor_f",
    ]
    chrono_xvar_list_jv = ["spincoat0", "pl_intensity_0", "pl_peakev_0", "pl_fwhm_0"]

    if jvdir_0 == None:
        for yvar in chrono_yvar_list_pl:
            correlation_plot(
                metricdf=metricdf,
                x_col="spincoat0",
                y_col=yvar,
                batch=batch,
                save=save,
            )
    if jvdir_0 != None:
        for yvar in chrono_yvar_list_pl:
            correlation_plot(
                metricdf=metricdf,
                x_col="spincoat0",
                y_col=yvar,
                batch=batch,
                save=save,
            )
        for xvar in chrono_xvar_list_jv:
            for yvar in chrono_yvar_list_jv:
                try:
                    correlation_plot(
                        metricdf=metricdf_dropped,
                        x_col=xvar,
                        y_col=yvar,
                        batch=batch,
                        save=save,
                    )
                except:
                    pass
        correlation_matrix_jv(
            metricdf=metricdf_dropped, method="pearson", batch=batch, save=save
        )
    if jvdir_0 == None:
        correlation_matrix(metricdf=metricdf, method="pearson", batch=batch, save=save)
    plot_df(rawdf, batch=batch, save=save)
    plot_bf(rawdf, batch=batch, save=save)
    plot_pl(rawdf, batch=batch, save=save)
    if save == True:
        os.chdir("..")
    return metricdf, rawdf


def plot_hist(
    data,
    x_var_list,
    hue_var_list,
    y_var_list,
    pce_lim=None,
    ff_lim=None,
    voc_lim=None,
    jsc_lim=None,
    rsh_lim=None,
    rs_lim=None,
    rch_lim=None,
    i_factor_lim=None,
    pl_intensity_lim=None,
    pl_fwhm_lim=None,
    pl_peakev_lim=None,
    psk_peak_lim=None,
    pskpbi2_ratio_lim=None,
):
    warnings.filterwarnings("ignore")

    horiz = len(y_var_list)
    vert = len(x_var_list)
    embiggen = 4
    color_list_subset = ["red", "blue", "green"]

    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=True,
    )

    for k in range(vert):
        for n in range(horiz):
            y_var = y_var_list[n]

            x_var = x_var_list[k]
            hue_var = hue_var_list[k]

            ax[k, n] = sns.boxplot(
                x=x_var,
                y=y_var,
                hue=hue_var,
                data=data,
                palette="Set1",
                width=0.5,
                showfliers=False,
                ax=ax[k, n],
            )
            ax[k, n].get_legend().remove()
            ax[k, n].legend(
                title=hue_var,
                labels=sorted(data[hue_var].unique()),
                labelcolor=color_list_subset,
            )

            ax[k, n] = sns.stripplot(
                x=x_var,
                y=y_var,
                hue=hue_var,
                data=data,
                palette="Set1",
                dodge=True,
                edgecolor="black",
                linewidth=0.5,
                size=3,
                ax=ax[k, n],
            )
            ax[k, n].get_legend().remove()
            ax[k, n].legend(
                title=hue_var,
                labels=sorted(data[hue_var].unique()),
                labelcolor=color_list_subset,
            )

            if x_var == "all":
                ax[k, n].set_xlabel("")

            if y_var == "pce" or y_var == "pce_f" or y_var == "pce_r":
                y_axis_label = "Power Conversion Effiency %"
                ax[k, n].set_ylabel(y_axis_label)
                if pce_lim:
                    ax[k, n].set(ylim=(pce_lim[0], pce_lim[1]))

            if y_var == "jsc" or y_var == "jsc_f" or y_var == "jsc_r":
                y_axis_label = "J$_{SC}$ mA/cm$^2$"
                ax[k, n].set_ylabel(y_axis_label)
                if jsc_lim:
                    ax[k, n].set(ylim=(jsc_lim[0], jsc_lim[1]))

            if y_var == "voc" or y_var == "voc_f" or y_var == "voc_r":
                y_axis_label = "V$_{OC}$ mV"
                ax[k, n].set_ylabel(y_axis_label)
                if voc_lim:
                    ax[k, n].set(ylim=(voc_lim[0], voc_lim[1]))

            if y_var == "ff" or y_var == "ff_f" or y_var == "ff_r":
                y_axis_label = "Fill Factor %"
                ax[k, n].set_ylabel(y_axis_label)
                if ff_lim:
                    ax[k, n].set(ylim=(ff_lim[0], ff_lim[1]))

            if y_var == "rsh" or y_var == "rsh_f" or y_var == "rsh_r":
                y_axis_label = "Shunt Resistance Ωcm$^2$"
                ax[k, n].set_ylabel(y_axis_label)
                if rsh_lim:
                    ax[k, n].set(ylim=(rsh_lim[0], rsh_lim[1]))

            if y_var == "rs" or y_var == "rs_f" or y_var == "rs_r":
                y_axis_label = "Series Resistance Ωcm$^2$"
                ax[k, n].set_ylabel(y_axis_label)
                if rs_lim:
                    ax[k, n].set(ylim=(rs_lim[0], rs_lim[1]))

            if y_var == "rch" or y_var == "rch_f" or y_var == "rch_r":
                y_axis_label = "Characteristic Resistance Ωcm$^2$"
                ax[k, n].set_ylabel(y_axis_label)
                if rch_lim:
                    ax[k, n].set(ylim=(rch_lim[0], rch_lim[1]))

            if y_var == "i_factor" or y_var == "i_factor_f" or y_var == "i_factor_r":
                y_axis_label = "Ideality Factor"

                ax[k, n].set_ylabel(y_axis_label)
                if i_factor_lim:
                    ax[k, n].set(ylim=(i_factor_lim[0], i_factor_lim[1]))

            if y_var == "pl_intensity_0":
                y_axis_label = "PL Intensity (counts/second)"
                ax[k, n].set_ylabel(y_axis_label)
                if pl_intensity_lim:
                    ax[k, n].set(ylim=(pl_intensity_lim[0], pl_intensity_lim[1]))

            if y_var == "pl_fwhm_0":
                y_axis_label = "PL FWHM (eV)"
                ax[k, n].set_ylabel(y_axis_label)
                if pl_fwhm_lim:
                    ax[k, n].set(ylim=(pl_fwhm_lim[0], pl_fwhm_lim[1]))

            if y_var == "pl_peakev_0":
                y_axis_label = "PL Peak Energy (eV)"
                ax[k, n].set_ylabel(y_axis_label)
                if pl_peakev_lim:
                    ax[k, n].set(ylim=(pl_peakev_lim[0], pl_peakev_lim[1]))

            if y_var == "pl_peakev_0":
                y_axis_label = "PL Peak Energy (eV)"
                ax[k, n].set_ylabel(y_axis_label)
                if pl_peakev_lim:
                    ax[k, n].set(ylim=(pl_peakev_lim[0], pl_peakev_lim[1]))

            if y_var == "psk_peak_intensity":
                y_axis_label = "Perovskite <100> Intensity (counts)"
                ax[k, n].set_ylabel(y_axis_label)
                if psk_peak_lim:
                    ax[k, n].set(ylim=(psk_peak_lim[0], psk_peak_lim[1]))

            if y_var == "psk_to_pbi2":
                y_axis_label = "Perovskite <100>:PbI$_2$ <100> Ratio"
                ax[k, n].set_ylabel(y_axis_label)
                if psk_peak_lim:
                    ax[k, n].set(ylim=(pskpbi2_ratio_lim[0], pskpbi2_ratio_lim[1]))

    warnings.filterwarnings("default")
    plt.show()


def plot_heatmaps(
    data, metric_display_choice, metric_var_list, round_val=2, map_method="cubic"
):
    if metric_display_choice == "median":
        z_var_list = [x + "_median" for x in metric_var_list]

    std_var_list = [x + "_std" for x in metric_var_list]

    # fig, ax = plt.subplots(figsize=(4,4))
    # objs = ['literallyanything' for i in range(1)]
    warnings.filterwarnings("ignore")
    horiz = len(z_var_list)
    vert = 2
    embiggen = 3
    fig, ax = plt.subplots(
        vert,
        horiz,
        figsize=(horiz * embiggen, vert * embiggen),
        constrained_layout=True,
    )
    for k in range(vert):
        for n in range((horiz)):

            x_name = x_var_list[k]
            y_name = y_var_list[k]
            z_name = z_var_list[n]
            std_name = std_var_list[n]

            x = data[x_name]
            y = data[y_name]
            z = data[z_name]
            z_std = data[std_name]

            # median
            xy = pd.concat([x, y], axis=1)
            xyz = pd.concat([xy, z], axis=1)
            xyz = xyz.dropna()
            x = xyz[x_name].astype(float)
            y = xyz[y_name].astype(float)
            z = xyz[z_name].astype(float)
            x_len = len(x)
            x_grid = sorted(df_metric_all[x_var_list[k]].unique())
            y_grid = sorted(df_metric_all[y_var_list[k]].unique())
            X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
            grid = np.meshgrid(x_grid, y_grid)
            data_subset = np.stack((x, y), axis=-1)
            z_array = np.array(z)
            map = scipy.interpolate.griddata(
                data_subset,
                z_array,
                (grid[0], grid[1]),
                method=map_method,
                rescale=False,
                fill_value=np.nan,
            )
            labels = map.round(round_val).astype(str).astype(object)

            # std
            xy_std = pd.concat([x, y], axis=1)
            xystd = pd.concat([xy_std, z_std], axis=1)
            xystd = xystd.dropna()
            x_std = xystd[x_name].astype(float)
            y_std = xystd[y_name].astype(float)
            z_std = xystd[std_name].astype(float)
            x_len = len(x)

            data_subset_std = np.stack((x_std, y_std), axis=-1)
            z_array_std = np.array(z_std)
            try:
                map_std = scipy.interpolate.griddata(
                    data_subset_std,
                    z_array_std,
                    (grid[0], grid[1]),
                    method=map_method,
                    rescale=False,
                    fill_value=np.nan,
                )
                labels_std = map_std.round(round_val).astype(str).astype(object)
            except:
                map_std = np.empty(map.shape)
                map_std[:] = np.nan
                labels_std = map_std.round(round_val).astype(str).astype(object)

            cm1 = mpl.cm.get_cmap("coolwarm")  # divering color
            cm2 = mpl.cm.get_cmap("viridis")  # linear color

            cm1.set_bad(color="black")  # sets color of missing data
            cm2.set_bad(color="grey")  # sets color of missing data

            cmap_choice = cm2

            label_median_and_std = labels + "\n" + labels_std
            im = sns.heatmap(
                map,
                cmap="viridis",
                vmin=np.nanmin(map),
                vmax=np.nanmax(map),
                annot=label_median_and_std,
                cbar=True,
                ax=ax[k, n],
                fmt="",
                cbar_kws={"label": f"{z_name}", "aspect": 100},
                annot_kws={"size": 6},
            )

            # objs[0]= plt.colorbar(im, ax = ax, label = f'pce_r', fraction = 0.046)

            ax[k, n].set_xticks(np.linspace(0.5, 2.5, 3))
            ax[k, n].set_xticklabels(x_grid)

            ax[k, n].set_yticks(np.linspace(0.5, 2.5, 3))
            ax[k, n].set_yticklabels(np.array(y_grid))

            ax[k, n].set_xlabel(x_var_list[k])
            ax[k, n].set_ylabel(y_var_list[k])
    warnings.filterwarnings("default")

    plt.show()
