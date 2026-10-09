import pathlib
from functools import lru_cache

from dotenv import find_dotenv as find_file

BASE = pathlib.Path(find_file("pyproject.toml")).parent.resolve()

FNAME_GLODAP_SURF = BASE / "data/raw/GLODAPv2023/GLODAPv2.2023_surface.pq"


@lru_cache(2)
def load_parquet_file(fname, time_name="time"):
    import polars as pl

    print(f"Loading data for the first time {fname}")
    df = pl.read_parquet(fname)
    df = df.with_columns(
        df["time"].cast(pl.datatypes.Datetime(time_unit="ns")).alias("time"),
        df["time"].dt.year().alias("year"),
    )

    return df


def load_glodap(year: int | None = None, fname: pathlib.Path = FNAME_GLODAP_SURF, time_name="time"):
    import polars as pl

    df = load_parquet_file(fname, time_name=time_name)

    if year is not None:
        df = df.filter(pl.col("year") == year)

    return df.to_pandas()
