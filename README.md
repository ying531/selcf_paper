# selcf_paper
This repository contains the Python code to reproduce the results in Selection by Prediction paper.



The simulation in the paper was run with Python 3. The following Python packages are required to be installed: `numpy`, `pandas`, `sklearn`.



## Folders 

- `simulations/`: bash file for running the simulations in batch.
- `utils/`: Python codes for the simulations. 
- `results/`: store all the experiment outputs, will be automatically created if this directory does not exist.



## Running simulations

#### Single run 

Calling the file `simu.py` executes one run of the simulation. It takes five inputs: `--sig` from 1 to 10 corresponds to the noise strength $\sigma$ in the paper from 0.1 to 1), `--nt_id` from 1 to 4 corresponds to test sample sizes 10, 100, 500, 1000, `--set_id` from 1 to 8 corresponds to the eight data generating processes in the paper (Table 2), `--q` from 1, 2, 5 corresponds to FDR level 0.1, 0.2 and 0.5, `--seed` from 1 to 1000 is the random seed used in this run. 



It iterates over all the three machine learning algorithm (`gbr`, `rf` and `svm`) in the paper and three nonconformity scores (`BH_res`, `BH_rel`, `BH_clip`) in one single run. 



For example, to execute a single run of the experiment for noise strength 0.4, test sample size 100 in setting 7, with FDR level 0.1 and random seed 53, simple run the following script:

```bash
cd simulations 
python3 simu.py 4 2 7 1 53
```



#### Batch submission 

The simulations can also be submitted in a batch mode on computing clusters, using the bash file in `bash/` folder (may need modification according to the configurations of the computing clusters). 

The curret bash file runs `--sig` from 1 to 10, `--nt_id` from 1 to 4, `--q` in {1,2,5}, and `--seed` from 1 to 100. To submit these jobs, direct to `bash/` folder and run 

```bash
sh bash.sh
```

These parameters can be edited. 