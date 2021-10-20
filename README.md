# Adversarial Attacks on ML-based Intrusion Detection
This is the official code release of the following paper: 

Machine learning is a widely used methodology for anomaly detection in most
networked systems. Despite their high performance, deep neural networks are vulnerable through semantic modification of the inputs (e.g., adversarial examples),
posing significant risks for security-critical applications. So far, adversarial machine
learning is primarily studied in computer vision and is still emerging in other data
domains. In this dissertation, the robustness of ML-based network intrusion detection systems is evaluated by perturbing the input samples at test time (evasion
attacks). The main contribution is maintaining the test samples’ imperceptibility, functionality, and input coherence while evading a trained model. In pursuit
of this objective, two state-of-the-art gradient-based algorithms are extended in a
constrained domain to generate realistic attacks against a classifier for intrusion detection. The investigations show that the narrow threat surface is still sufficiently
large for an adversary, and constraints do not make a domain resilient.

<img src="https://github.com/hsekol-hub/Adversarial-Attacks-ML-based-IDS/blob/main/config/tmp/pipeline.png" alt="pipeline" width="700" class="center">

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
Once the raw datasets are in data/<data-set name>
Update the flags defined in run.py to perform the preprocessing steps. 

```
cd core
python run.py --dataset <name> --preprocess_data True -- sample_data False
```

### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other experiment set up according to 
Section 5.1.4 in the paper. 
The configurations are defined in two .YAML files 
1. config/ip_config.yaml
2. ip_perturb_config.yaml

### Evaluate models
Then the following commands can be used to train the proposed models and craft adversarial examples.
1. Once the run.py file is executed the console promots user to train a specific ML model.
2. Similarly attack algorithms are prompted.


## Citation
If you find the resource in this repository helpful, please cite TBA
