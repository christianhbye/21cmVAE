.. image:: https://zenodo.org/badge/360315069.svg
   :target: https://zenodo.org/badge/latestdoi/360315069
   
.. image:: https://readthedocs.org/projects/21cmvae/badge/?version=latest
   :target: https://21cmvae.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**********************
Very Accurate Emulator
**********************

Very Accurate Emulator (21cmVAE) is an emulator of the 21-cm global signal. Given an input of seven astrophyscial parameters, it directly computes realizations of the global signal across redshifts 5-50. The emulator is described in detail in [Bye et. al, 2021](https://arxiv.org/abs/2107.05581). 

21cmVAE emulates global signals with an average error of 0.34% of the signal amplitude (equivalently 0.54 mK) with a run time of 40 ms on average. It is trained on ~30,000 cases, the same training set as the other exisiting emulators [21cmGEM](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract) and [globalemu](https://ui.adsabs.harvard.edu/abs/2021arXiv210404336B/abstract). The accuracy and speed makes 21cmVAE a possible tool for parameter fitting, using samplig techinques like MCMC. Moreover, the variational autoencoder approach creates an interpretable latent space that allows us to determine the relative importance of the model parameters on the global 21-cm signal. 

21cmVAE is free to use on the MIT open source license. We provide here our best pretrained model, as well as code to modify and train new models. We also provide the code used for the hyperparameter tuner and code to run and train the direct emulator. We refer to the sample notebooks for an introduction on how to run and test the pretrained model, as well as how to train new models. 

Questions and comments are welcome; please e-mail me at chbye@berkeley.edu. If you use this work for academic purposes, please cite [Bye et. al, 2021](https://arxiv.org/abs/2107.05581) and link to this repository.

Set-up
######

Dependencies: Python 3, Tensorflow 2, h5py, numpy.
Recommended: Matplotlib (required for the sample notebooks).

The simplest way to install 21cmVAE with all dependencies is with pip:
.. code:: bash
   python -m pip install 21cmVAE

Alternatively, you may clone the Git repository:

.. code:: bash

   git clone https://github.com/christianhbye/21cmVAE.git
   python setup.py --install

Finally, download the dataset used from http://doi.org/10.5281/zenodo.5084114, and move the file to the 21cmVAE folder. This is necessary for all uses of the emulator, as the dataset is used in the prediction alogrithm.
