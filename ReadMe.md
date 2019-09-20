This is my code to (locally) sample the posterior distribution of cosmological and nuisscance parameters given Supernova and Quasar data. With minor modifications, this can be run on a cluster

To run the notebook Sample_Cosmo.ipynb, you will need the package emcee (version 3.0.0). Unfortunately, "pip install emcee" or "conda install -c conda-forge emcee" will install an older version that requires some hacks and the autocorrelation time cannot be calculated. Therefore, the best way to run this is to execute the following steps using anaconda:

1) execute git clone https://github.com/MoPl90/cosmological_tests in your preferred directory

2) conda create -n emcee3 numpy scipy jupyter matplotlib astropy tqdm h5py schwimmbad mpi4py

3) conda activate emcee3

4) conda install -c conda-forge corner

5) git clone https://github.com/dfm/emcee.git

6) cd emcee

7) python setup.py install

8) cd ../cosmological_tests

9) jupyter-notebook Sample_Cosmo.ipynb *or* mpirun -np [numebr of prallel processes] python sample.py [data sets to include] [cosmological model]

Now the Code should run!


The Supernova data is from the Joint Light curve analysis (Betoule et al., arXiv:1401.4064)

Qusar data courtesy of Elisabeta Risaliti; analysis following (Risaliti & Lusso,  arXiv:1505.07118 and arXiv:1811.02590)

BAO data: see publication.

CMB data: Planck legacy archive (https://pla.esac.esa.int/)