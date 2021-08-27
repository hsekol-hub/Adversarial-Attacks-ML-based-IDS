# Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning

This is the official code release of the following paper: 

TBA

<img src="https://github.com/hsekol-hub/Adversarial-Attacks-ML-based-IDS/blob/main/config/tmp/pipeline.png" alt="pipeline" width="700" class="center">

## Quick Start

### Environment variables & installations
First, clone repository
Install virtualenv
```
pip install virtualenv

virtualenv -p python3.7 venv

source venv/bin/activate
```
### Install dependencies
```
pip install -r requirement.txt
```

### Process data
First, unzip and unpack the data files 
```
tar -zxvf data-release.tar.gz
```
For the three ICEWS datasets `ICEWS18`, `ICEWS14`, `ICEWS05-15`, go into the dataset folder in the `./data` directory and run the following command to construct the static graph.
```
cd ./data/<dataset>
python ent2word.py
```

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.


### Evaluate models


### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other experiment set up according to Section 5.1.4 in the paper  


## Citation
If you find the resource in this repository helpful, please cite