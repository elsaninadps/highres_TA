# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: highres_TA (3.11.14)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import highres_ta as ta
import dask.dataframe as dd
from model_selection import load_data, normalize_alkalinity

# %%
data = load_data()
data['alk_normed'] = normalize_alkalinity(data.talk, data.salinity, 34.5)
data

# %%
data.talk.plot.hist(bins=30)
data.talk.describe()

# %%
data.alk_normed.plot.hist(bins=30)
data.alk_normed.describe()

# %%
data.salinity.plot.hist(bins=30)
data.salinity.describe()

# %%
mask = (
    ((data.salinity < 20) | (data.salinity > 40))
)
data[~mask].alk_normed.plot.hist(bins=30)
data[~mask].alk_normed.describe()
# %%
