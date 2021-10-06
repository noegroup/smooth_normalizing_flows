
# Smooth Normalizing Flows

Source code for

J. Köhler, A. Krämer, and F. Noé.  Smooth Normalizing Flows.  arXiv preprint [arXiv:2110.00351](https://arxiv.org/abs/2110.00351).


## Installation
- `conda env create -f condaenv.yml`
- `conda activate smooth_normalizing_flows`
- `pip install nflows`
- Download and install [bgflow](https://github.com/noegroup/bgflow)
    - ```
      git clone git@github.com:noegroup/bgflow.git
      cd bgflow
      git checkout factory-compact-interval
      python setup.py install
      ```
- Download and install [bgmol](https://github.com/noegroup/bgmol) (same as above)
    - ```
      git clone git@github.com:noegroup/bgmol.git
      cd bgmol
      python setup.py install
      ```
- Install the alanine dipeptide model `cd ala2flow  && python setup.py install && cd -`


## Experiments


### 2D Toy Examples
See notebooks in the `experiment_toy2d` directory.


### Alanine Dipeptide examples
Training is done via the `train.py` script in the `experiment_ala2` directory.
Call `python train.py --help` to see the available options.

The plotting and analysis uses checkpoint files written by the train.py script.
In each notebook, the `CHECKPOINT` variable has to be set to the checkpoint path.

MD simulations with flows are run with the `simulate.ipynb` notebook in the `experiment_ala2` directory.

### Runtime of Multi-Bin Biscection
To generate the commands for all tests:
```
cd experiment_runtime
python submit.py 
```
