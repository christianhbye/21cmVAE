.. image:: https://zenodo.org/badge/360315069.svg
   :target: https://zenodo.org/badge/latestdoi/360315069
 

**********************
Very Accurate Emulator
**********************

Very Accurate Emulator (21cmVAE) is an emulator of the 21-cm global signal. Given an input of seven astrophyscial parameters, it directly computes realizations of the global signal across redshifts 5-50. The emulator is described in detail in `Bye et. al 2022 <https://iopscience.iop.org/article/10.3847/1538-4357/ac6424>`__.

21cmVAE emulates global signals with an average error of 0.34% of the signal amplitude (equivalently 0.54 mK) with a run time of 40 ms on average. It is trained on ~30,000 cases, the same training set as the other exisiting emulators `21cmGEM <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract>`_ and `globalemu <https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract>`_. The accuracy and speed makes 21cmVAE a possible tool for parameter fitting, using samplig techinques like MCMC. Moreover, the variational autoencoder approach creates an interpretable latent space that allows us to determine the relative importance of the model parameters on the global 21-cm signal. 

21cmVAE is free to use on the MIT open source license. We provide here our best pretrained model, as well as code to modify and train new models. We also provide the code used for the hyperparameter tuner and code to run and train the direct emulator. We refer to the sample notebooks for an introduction on how to run and test the pretrained model, as well as how to train new models. 

Questions and comments are welcome; please e-mail me at chb@berkeley.edu. If you use this work for academic purposes, please cite `Bye et. al 2022 <https://iopscience.iop.org/article/10.3847/1538-4357/ac6424>`__ and link to this repository.

Set-up
######

Dependencies: python>=3.6, tensorflow>=2.5, h5py, jupyter, matplotlib, numpy, tqdm

For developing: black, flake8, pytest, tox

To install 21cmVAE in a `virtual environment <https://docs.python.org/3/library/venv.html>`_ (recommended) with all required dependencies, do:

.. code:: bash

   git clone https://github.com/christianhbye/21cmVAE.git
   cd 21cmVAE
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install .

For development, please install with :code:`python -m pip install .[dev]` to get the extra dependencies.

Versioning
##########
21cmVAE uses `semantic versoning <https://semver.org/>`_.

Contributions
#############
Main author: Christian H. Bye

Suggestions and additional contributions from:

- Stephen KN Portillo

- Anastasia Fialkov

If you have suggestions for improvements/additional features, notice a bug, or want to contribute in another way, please open an issue, make a pull request or just e-mail me (chb@berkeley.edu).
