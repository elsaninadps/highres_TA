import numpy as np
import pandas as pd


def latlon_to_spherical_coords(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    """
    Convert latitude and longitude to spherical coordinates (x, y, z).

    Parameters:
    - lat: pd.Series of latitudes in degrees.
    - lon: pd.Series of longitudes in degrees.

    Returns:
    - Tuple of three pd.Series: (x, y, z) coordinates.
    """
    # Convert degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate spherical coordinates
    n_coords = pd.DataFrame({
        "ncoord_x": pd.Series(np.cos(lat_rad) * np.cos(lon_rad), name="ncoord_x"),
        "ncoord_y": pd.Series(np.cos(lat_rad) * np.sin(lon_rad), name="ncoord_y"),
        "ncoord_z": pd.Series(np.sin(lat_rad), name="ncoord_z"),
    })

    return n_coords


def time_to_cyclical_dayofyear(time: pd.Series) -> pd.DataFrame:
    """
    Convert time to cyclical day-of-year features (sin and cos).

    Parameters:
    - time: pd.Series of datetime objects.

    Returns:
    - pd.DataFrame with two columns: 'dayofyear_sin' and 'dayofyear_cos'.
    """
    # Extract day of year
    day_of_year = time.dt.dayofyear

    # Calculate sine and cosine transformations
    dayofyear_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    dayofyear_cos = np.cos(2 * np.pi * day_of_year / 365.25)

    return pd.DataFrame({"dayofyear_sin": dayofyear_sin, "dayofyear_cos": dayofyear_cos})


def add_spherical_coords(
    df: pd.DataFrame, lat_col_name: str = "lat", lon_col_name: str = "lon"
) -> pd.DataFrame:
    """
    Add spherical coordinates (x, y, z) to the DataFrame based on latitude and longitude.

    Parameters:
    - df: pd.DataFrame containing the data.
    - lat_col_name: Name of the column in df that contains latitude values.
    - lon_col_name: Name of the column in df that contains longitude values.

    Returns:
    - pd.DataFrame with additional columns 'ncoord_x', 'ncoord_y', and 'ncoord_z'.
    """
    df_all = df.index.to_frame()
    spherical_coords = latlon_to_spherical_coords(
        df_all[lat_col_name], df_all[lon_col_name]
    ).set_index(df.index)
    return df.assign(**spherical_coords)


def add_cyclical_dayofyear(df: pd.DataFrame, time_col_name: str = "time") -> pd.DataFrame:
    """
    Add cyclical day-of-year features (sin and cos) to the DataFrame based on a time column.

    Parameters:
    - df: pd.DataFrame containing the data.
    - time_col_name: Name of the column in df that contains datetime values.

    Returns:
    - pd.DataFrame with additional columns 'dayofyear_sin' and 'dayofyear_cos'.
    """
    df_all = df.index.to_frame()
    cyclical_features = time_to_cyclical_dayofyear(df_all[time_col_name]).set_index(df.index)

    return df.assign(**cyclical_features)
