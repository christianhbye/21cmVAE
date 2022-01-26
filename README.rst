.. image:: https://zenodo.org/badge/360315069.svg
   :target: https://zenodo.org/badge/latestdoi/360315069
 

**********************
Very Accurate Emulator
**********************

Very Accurate Emulator (21cmVAE) is an emulator of the 21-cm global signal. Given an input of seven astrophyscial parameters, it directly computes realizations of the global signal across redshifts 5-50. The emulator is described in detail in `Bye et. al, 2022 <https://arxiv.org/abs/2107.05581>`__.

21cmVAE emulates global signals with an average error of 0.34% of the signal amplitude (equivalently 0.54 mK) with a run time of 40 ms on average. It is trained on ~30,000 cases, the same training set as the other exisiting emulators `21cmGEM <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract>`_ and `globalemu <https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2923B/abstract>`_. The accuracy and speed makes 21cmVAE a possible tool for parameter fitting, using samplig techinques like MCMC. Moreover, the variational autoencoder approach creates an interpretable latent space that allows us to determine the relative importance of the model parameters on the global 21-cm signal. 

21cmVAE is free to use on the MIT open source license. We provide here our best pretrained model, as well as code to modify and train new models. We also provide the code used for the hyperparameter tuner and code to run and train the direct emulator. We refer to the sample notebooks for an introduction on how to run and test the pretrained model, as well as how to train new models. 

Questions and comments are welcome; please e-mail me at chbye@berkeley.edu. If you use this work for academic purposes, please cite `Bye et. al, 2022 <https://arxiv.org/abs/2107.05581>`__ and link to this repository.

Set-up
######

Dependencies: Python>=3.6, Tensorflow>=2.5, h5py, numpy.

Optional: matplotlib, jupyter (both required for the sample notebooks).



We recommend installing 21cmVAE in a `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_. To set up one with all the required dependencies for 21cmVAE, do:

.. code:: bash

   git clone https://github.com/christianhbye/21cmVAE.git
   cd 21cmVAE
   conda create env --prefix emulator_env -f environment.yml
   python -m pip install .

After installing 21cmVAE, download the dataset used from http://doi.org/10.5281/zenodo.5084114, and move the file to the VeryAccurateEmulator folder. This is necessary for all uses of the emulator, as the dataset is used in the prediction alogrithm.

Contributions
#############
Main author: Christian H. Bye

Suggestions and additional contributions from:

- Stephen KN Portillo

- Anastasia Fialkov

If you have suggestions for improvements/additional features, notice a bug, or want to contribute in another way, please open an issue, make a pull request or just e-mail me (chb@berkeley.edu).
