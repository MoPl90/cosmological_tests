This is my code to (locally) sample the posterior distribution of cosmological and nuisscance parameters given Supernova and Quasar data. With minor modifications, this can be run on a cluster

To run the notebook Sample_Cosmo.ipynb, you will need the package emcee (version 3.0.0). Unfortunately, "pip install emcee" or "conda install -c conda-forge emcee" will install an older version that requires some hacks and the autocorrelation time cannot be calculated. Therefore, the best way to run this is to execute the following steps using anaconda:

1) conda create -n emcee3 numpy scipy matplotlib astropy tqdm h5py

2) conda activate emcee3

3) conda install -c conda-forge corner

4) git clone https://github.com/dfm/emcee.git

5) cd emcee

6) python setup.py install

7) jupyter-notebook Sample_Cosmo.ipynb

Now the Code should run!


The Supernova data is from the Joint Light curve analysis (Betoule et al., arXiv:1401.4064)

Qusar data courtesy of Elisabetta Risaliti; analysis following (Risaliti & Lusso,  arXiv:1505.07118 and arXiv:1811.02590)