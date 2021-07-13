[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5085445.svg)](https://doi.org/10.5281/zenodo.5085445)

# Very Accurate Emulator

Very Accurate Emulator (21cmVAE) is an emulator of the 21-cm global signal. It is based on a variational autoencoder, which creates a low-dimensional representation of the global signal, and a neural network that predicts the autoencoder representation given an input of seven astrophyscial parameters. The emulator is described in detail in [Bye et. al, 2021](https://arxiv.org/abs/2107.05581). 

21cmVAE emulates global signals with an average error of 0.41% of the signal amplitude (equivalently 0.66 mK) with a run time of 40 ms on average. It is trained on ~30,000 cases, the same training set as the other exisiting emulators [21cmGEM](https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4845C/abstract) and [globalemu](https://ui.adsabs.harvard.edu/abs/2021arXiv210404336B/abstract). The accuracy and speed makes 21cmVAE a possible tool for parameter fitting, using samplig techinques like MCMC. Moreover, the variational autoencoder approach creates an interpretable latent space that allows us to determine the relative importance of the model parameters on the global 21-cm signal. 

21cmVAE is free to use on the MIT open source license. We provide here our best pretrained model, as well as code to modify and train new models. We also provide the code used for the hyperparameter tuner and code to run and train the direct emulator. We refer to the sample notebook for an introduction on how to run and test the pretrained model, as well as how to train new models. 

Questions and comments are welcome; please e-mail me at chbye@berkeley.edu. If you use this work for academic purposes, please cite [Bye et. al, 2021](https://arxiv.org/abs/2107.05581) and link to this repository.

## Set-up
Dependencies: Python 3.7.10, Tensorflow 2.5.0, h5py, numpy. In order to load the pretrained VAE, these versions are needed. If you do not wish to load the VAE, any version of Python 3 and Tensorflow 2 should work.

To install, make sure you have the dependencies. Use for example a virtual environment:
```
conda create -n emulator_env ipykernel python=3.7.10 tensorflow=2.5.0 numpy h5py
conda activate emulator_env
python -m ipykernel install --user --name emulator_env --display-name "21cmVAE"
```
Then, clone the repository:
```
git clone https://github.com/christianhbye/21cmVAE.git
```

Finally, download the dataset used from http://doi.org/10.5281/zenodo.5084114, and move the file to the 21cmVAE folder. This is necessary for all uses of the emulator, as the dataset is used in the prediction alogrithm.
