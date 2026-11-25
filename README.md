# HiggsML Package

[![Documentation Status](https://readthedocs.org/projects/black-swan-pkg/badge/?version=latest)](https://black-swan-pkg.readthedocs.io/en/latest/?badge=latest)

## Installation

```shell
pip install HiggsML
```
## Modules
The Package consists of 5 modules :

* 1 `ingestion`: This module contains the `Ingestion` class which takes care of the ingestion process of the competition. i.e. loading data into the model, initialization of the model, prediction etc. (`ingestion_parallel.py` is the parallel version of the ingestion module which can be used to speed up the ingestion process.)
* 2 `datasets` This module contains the `Data` class which has the train and test data in required formats, It loads the data according to the file format. It also contains function to make pseudo experiments. 
* `derived_quantities` This module contains functions which has methods to calculate derived quantities.
* `score` This module contains the `Score` class which has methods to calculate the score of the model with *RMSE*, *MAE* and *Quantile Score*.

* 3 `systematics` This module has functions to add systematics to the data with based in Nuisance parameter like 
    * Tau Hadron Energy scale
    * Jet Energy Scale 
    * Soft MET
    * ttbar Background Normalisation
    * Di-Bososn Background Normalisation
    * Overall Background Normalisation 
* 4 `visualisation` This module contains the `Dataset_visualise` class which has methods to help visualise the data 
*  `run_competition` is the cli command to run the competition. 

## Usage
To run the competition, you can use the following command:
```bash
run_competition --help
```
