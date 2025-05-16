# HiggsML BlackSwan Package

[![Documentation Status](https://readthedocs.org/projects/black-swan-pkg/badge/?version=latest)](https://black-swan-pkg.readthedocs.io/en/latest/?badge=latest)

you can install this package by 
```shell
pip install HiggsML
```

The Package consists of 5 modules :

* 1 `ingestion`: This module contains the `Ingestion` class which takes case of the ingestion process of the competition. i.e. loading data into the model, initialization of the model, prediction etc. 
* 2 `datasets` This module contains the `Data` class which has the train and test data in required formats, It loads the data according to the file format. it also contains function to make pseudo experiments. 
* 3 `systematics` This module has functions to add systematics to the data with based in Nuisance parameter like 
    * Tau Hadron Energy scale
    * Jet Energy Scale 
    * Soft MET
    * W Background Normalisation
    * Overall Background Normalisation 
* 4 `visualisation` This module contains the `Dataset_visualise` class which has methods to help visualise the data 
