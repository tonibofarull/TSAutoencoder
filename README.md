# Autoencoder

Implementation of a Dilated Convolutional Autoencoder for univariate Time Series.

<br>
<div align="center">
	<img src="utils/assets/model.png" width="80%"/>
</div>
<br>

## Requirements

Python 3.8 required. Some packages are incompatible with version 3.9. See [here](https://github.com/ray-project/ray/issues/11287).

Install dependencies using pip,

```
pip install -r requirements.txt
```

## Getting Started

Structure of the project,

```
.
├── data     
│   └── ElectricDevices
├── src
│   ├── configs
│   │   ├── arma.yaml
│   │   ├── arma5.yaml
│   │   └── config.yaml
│   ├── dataloader.py
│   ├── experiments
│   │   ├── exp1-shapley_value.ipynb
│   │   └── exp2-acc_cor.py
│   ├── interpretability.py
│   ├── main.ipynb
│   ├── models
│   │   ├── CAE.py
│   │   └── losses.py
│   ├── train.py
│   ├── tuning.py
│   └── utils.py
├── utils
└── weights
```

Execute the jupyter notebook `main.ipynb` to load the data, train the model and obtain the evaluation and interpretation.
