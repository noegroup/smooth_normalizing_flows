
# Smooth Normalizing Flows


## Install
-  Download data (follow instructions in ./bgmol/bgmol/data/README.md).
- `conda env create -f condaenv.yml`
- `conda activate smooth_normalizing_flows`
- `pip install nflows`
- `cd bgflow && python setup.py install && cd -`
- `cd bgmol  && python setup.py install && cd -`
- `cd bgforces  && python setup.py install && cd -`


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
