import pooch
import xarray as xr


def decompress_tar_gz(fname0, a=None, p=None):
    import pooch

    gz = pooch.Decompress()
    tar = pooch.Untar()

    fname1 = gz(fname0, a, p)
    fname2 = tar(fname1, a, p)

    return fname2


def get_glodap_mapped_product(prefix="../data/glodap/", space_res=0.25):
    from pathlib import Path as posixpath

    from ..gridded.regrid import resample_space

    seamask = (
        xr.open_dataset(
            pooch.retrieve(
                url=(
                    "https://github.com/RECCAP2-ocean/shared-resources"
                    "/raw/master/regions/RECCAP2_region_masks_all.nc"
                ),
                known_hash=None,
            )
        )
        .seamask.assign_coords(lon=lambda a: (a.lon - 180) % 360 - 180)
        .sortby(["lat", "lon"])
    )

    url = posixpath(
        "https://www.nodc.noaa.gov/archive/arc0107"
        "/0162565/1.1/data/0-data/mapped/"
        "GLODAPv2.2016b_MappedClimatologies.tar.gz"
    )

    flist = pooch.retrieve(
        url,
        None,
        fname=url.name,
        path=prefix,
        downloader=pooch.HTTPDownloader(progressbar=True),
        processor=decompress_tar_gz,
    )

    ds = []
    for key in ["TAlk", "TCO2", "Cant", "pHtsinsitutp", "OmegaA"]:
        fname = [f for f in flist if key in f][0]
        ds += (xr.open_mfdataset(fname)[key].isel(depth_surface=0),)

    ds = (
        xr.merge(ds)
        .assign_coords(lon=lambda a: (a.lon - 180) % 360 - 180)
        .sortby(["lat", "lon"])
        .load()
        .interpolate_na("lon", limit=360)
        .where(seamask)
    )

    ds = resample_space(ds, res_space=space_res)

    return ds
