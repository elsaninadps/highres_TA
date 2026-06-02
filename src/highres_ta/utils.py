from functools import lru_cache
import dotenv 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pooch

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent


def save_figs_to_pdf(figs: list[plt.Figure], filename="_.pdf", **props):
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(filename) as pdf:
        for fig in figs:
            props = dict(dpi=150, bbox_inches="tight") | props
            pdf.savefig(fig, **props)
            
@lru_cache(1)
def make_target_grid(res=0.25):
    import numpy as np

    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)
    return xr.Dataset(coords={"lat": lat, "lon": lon})

