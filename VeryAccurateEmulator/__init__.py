__version__ = "3.0.0"
__author__ = "Christian Hellum Bye"

from VeryAccurateEmulator import emulator
from VeryAccurateEmulator import preprocess

from pathlib import Path

HERE = __file__[: -len("__init__.py")]
if not Path(HERE + "dataset_21cmVAE.h5").exists():
    import requests

    print("Downloading dataset.")
    r = requests.get(
        "https://zenodo.org/record/5084114/files/dataset_21cmVAE.h5?download=1"
    )
    with open(HERE + "dataset_21cmVAE.h5", "wb") as f:
        f.write(r.content)
