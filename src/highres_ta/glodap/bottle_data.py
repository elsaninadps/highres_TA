import logging
import os
from pathlib import Path as posixpath

import numpy as np
import pandas as pd
import xarray as xr

GLODAP_YEARS = range(1982, 2021)
GLODAP_VARIABLES = {
    "salt_stacked": "salt_stacked",
    "mld_press_soda_fillog10": "mld_fillog10_soda342",
    "chl_globcolour_fillog10": "chl_fillog10_globcolour",
    "ssh_duacs": "ssh_duacs",
    "temp_oisst": "sst_oisst",
}


def run_default_setup():
    sname = posixpath("../data/glodap/GLODAPv2.2020_surface_matched_8D_25km.pq")

    if not sname.is_file():
        df = match_glodap_bottle_data_with_gridded()
        df.to_parquet(sname)


def _zero_fill(col):
    return col.astype(int).astype(str).str.zfill(2)


def _read_GLODAPv2019_alkalinity_adjustments(filename):
    from pandas import read_csv

    glodap2019_adjustments = read_csv(filename, skiprows=2, skipinitialspace=True)

    alkalinity_adjustments = glodap2019_adjustments.set_index("cruise_expocode").alkalinity_adj

    bad_values = alkalinity_adjustments < -100
    alkalinity_adjustments = alkalinity_adjustments.where(~bad_values)

    return alkalinity_adjustments.to_dict()


def alkalinity_adjustments(
    expocodes,
    fname="../data/raw/GLODAPv2021/GLODAPv2.2019_adjustments_last_updated_on_2021_06_21.csv",
):

    logging.info(f"[GLODAP]   Alkalinity adjustments imported from {fname} (manually downloaded)")
    adjust = _read_GLODAPv2019_alkalinity_adjustments(fname)

    df = pd.DataFrame(index=expocodes.index)
    for key in adjust:
        print(".", end="")
        i = key == expocodes
        df.loc[i, "talk_adjust"] = adjust[key]

    bins = [0, 0.1, 5, 10, 50]
    df["talk_strat"] = (
        pd.cut(df.talk_adjust.abs(), bins, labels=range(4))
        .astype(float)
        .replace(np.nan, 4)
        .astype(int)
    )

    return df


def get_glodap_bottle_data(
    prefix="../data/raw/GLODAPv2023/",
    url="https://glodap.info/glodap_files/v2.2023/GLODAPv2.2023_Merged_Master_File.mat",
    sname="GLODAPv2.2023_{subset}.pq",
    overwrite=False,
):
    import pooch
    from scipy.io import loadmat

    sname_all = posixpath(f"{prefix}/{sname}".format(subset="all"))
    sname_surf = posixpath(f"{prefix}/{sname}".format(subset="surface"))

    if sname_all.is_file() and sname_surf.is_file() and not overwrite:
        logging.info(f"[GLODAP]   Returning existing file names: {sname_all}, {sname_surf}")
        return str(sname_all), str(sname_surf)

    logging.info(f"[GLODAP]   Reading dataset from {url}")
    fname = pooch.retrieve(
        url,
        None,
        fname=posixpath(url).name,
        path=prefix,
        downloader=pooch.HTTPDownloader(progressbar=True),
    )
    raw = loadmat(fname)

    print("Reading data in from MATLAB")
    # gets keys from the MATLAB file and creates a dataframe
    data = {k[2:]: raw[k].squeeze() for k in raw.keys() if k.startswith("G2")}
    df = pd.DataFrame.from_dict(data)

    print("Fixing time")
    # remove nan coords and then create datetime
    time_cols = ["year", "month", "day", "hour", "minute"]
    df = df.dropna(subset=time_cols)
    df["time"] = pd.to_datetime(
        df[time_cols].apply(_zero_fill).sum(axis=1).astype(int).astype(str),
        format="%Y%m%d%H%M",
    )

    # get expocodes from the MATLAB file and propogate into dataframe
    expocodes = dict(
        zip(
            [int(s[0].squeeze()) for s in raw["expocodeno"]],
            [str(s[0].squeeze()) for s in raw["expocode"]],
            strict=True,
        )
    )
    df["expocode"] = df.cruise.astype(int).replace(expocodes)

    print("Starting Alkalinity adjustment/stratification")
    # create columns for stratification (additional QC columns)
    # adjust = alkalinity_adjustments(
    #     df.expocode,
    #     fname='../data/raw/GLODAPv2023/GLODAPv2.2019_adjustments_last_updated_on_2021_06_21.csv')
    # df = pd.concat([df, adjust], axis=1)

    # drop nans in TALK and then drop all columns that are only nans
    df = df.dropna(subset=["talk"])

    keep = [
        "expocode",
        "time",
        "latitude",
        "longitude",
        "bottomdepth",
        "maxsampdepth",
        "depth",
        "temperature",
        "salinity",
        "oxygen",
        "aou",
        "nitrate",
        "nitrite",
        "silicate",
        "phosphate",
        "tco2",
        "fco2",
        "phtsinsitutp",
        "talk",
        "talkf",
        "talkqc",
    ]

    surface_talk = ((df.depth < 30) & (df.latitude.abs() < 30)) | (
        (df.depth < 20) & (df.latitude.abs() >= 30)
    )

    df_surface = df.loc[surface_talk, keep].rename(columns=dict(latitude="lat", longitude="lon"))

    df[keep].to_parquet(sname_all)
    df_surface.to_parquet(sname_surf)

    logging.info(f"[GLODAP]   Data saved to {sname_all}")
    logging.info(f"[GLODAP]   Data saved to {sname_surf}")

    return sname_all, sname_surf


def get_surface_index(
    lat, depth, lat_boundary=30, low_lat_depth_threshold=10, high_lat_depth_threshold=15
):
    return ((depth < low_lat_depth_threshold) & (lat.abs() < lat_boundary)) | (
        (depth < high_lat_depth_threshold) & (lat.abs() >= lat_boundary)
    )


def match_glodap_bottle_data_with_gridded(
    fname="../data/glodap/GLODAPv2.2023_surface.pq",
    years=GLODAP_YEARS,
    gridded_path="../data/gridded/gridded_8D_25km/{year}/*.nc",
    gridded_variables=GLODAP_VARIABLES,
    woa18_path="../data/gridded/gridded_8D_25km/gridded_8D_25km_woa18.nc",
    verbose=False,
    save_name="../data/glodap/GLODAPv2.2023_surface_matched_8D_25km.pq",
):
    from pathlib import Path as posixpath

    import pandas as pd

    from ..utils import colocate_xdarray

    if isinstance(save_name, str):
        sname = posixpath(save_name)
        if sname.is_file():
            return str(sname)
    else:
        sname = None

    fname = posixpath(fname)
    fname = get_glodap_bottle_data(sname=fname.name, prefix=str(fname.parent))[0]

    df = pd.read_parquet(fname)
    df = df.loc[df.time.dt.year.isin(list(years))]

    for year in years:
        logging.info(f"[GLODAP]   {year}: Loading netCDF files and colocating with GLODAP")
        iy = df.time.dt.year == year
        if iy.sum() == 0:
            continue
        coords = dict(time=df.loc[iy, "time"], lat=df.loc[iy, "lat"], lon=df.loc[iy, "lon"])
        ds = xr.open_mfdataset(
            gridded_path.format(year=year),
            combine="nested",
            parallel=True,
            preprocess=lambda a: a.drop("dayofyear") if "dayofyear" in a else a,
        )

        for key in gridded_variables:
            if key in ds:
                df.loc[iy, key] = colocate_xdarray(ds[key], **coords, verbose=verbose)

    logging.info("[GLODAP]   Loading WOA18 climatology and colocating with GLODAP")
    coords = dict(dayofyear=df.time.dt.dayofyear, lat=df.lat, lon=df.lon)
    woa18 = xr.open_dataset(woa18_path)
    for key in woa18:
        df.loc[:, f"woa18_{key}"] = colocate_xdarray(woa18[key], **coords, verbose=verbose)

    if sname is not None:
        sname.parent.mkdir(exist_ok=True, parents=True)
        df.to_parquet(sname)
        return sname
    else:
        return df


def grid_1deg_monthly_glodap_surface_data(pq_fname, sname):
    from ...utils import save_nc4

    if os.path.exists(sname):
        return sname

    glodap = pd.read_parquet(pq_fname)
    glodap_alk = glodap

    glodap_alk["lon"] = (glodap_alk.lon.clip(-179.99, 179.99) // 1) + 0.5
    glodap_alk["lat"] = (glodap_alk.lat.clip(-89.99, 89.99) // 1) + 0.5
    glodap_alk["time"] = glodap_alk.time.astype("datetime64[M]")

    mask = glodap_alk.time >= "1982"
    glodap_alk = glodap_alk.loc[mask]
    glodap_alk = glodap_alk.groupby(["time", "lat", "lon"]).mean()
    glodap_alk["depth"] = glodap_alk.depth.round()
    glodap = (
        glodap_alk.to_xarray()
        .transpose("time", "lat", "lon")
        .reindex(
            time=pd.date_range("1982-01", "2020-12-31", freq="1MS"),
            lat=np.arange(-89.5, 90),
            lon=np.arange(-179.5, 180),
        )
    )

    save_nc4(glodap, sname)

    return sname
