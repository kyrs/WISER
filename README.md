# WISER
[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)
## Advancing cancer drug response prediction through weak supervision and supervised representation learning
## Acknowledgement
Current code base is based on 
1. https://github.com/XieResearchGroup/CODE-AE
2. https://github.com/hunterlang/weaksup-subset-selection  
## Architecture
![architecture](./images/arch.png?raw=true)
## Overview 
Our work introduces a novel representation learning approach that incorporates drug response information during the domain-invariant representation learning phase. We also utilize weak supervision aided by subset selection to efficiently predict drug responses, leveraging patient genomic profiles without documented drug response.
## Installation
1. Install anaconda:
Instructions here: https://www.anaconda.com/download/
2. pip install -r requirements.txt
3. Download benchmark datasets available at Zenodo [http://doi.org/10.5281/zenodo.4776448]   (version 2.0)
4. Replace the downloaded path in the config/data_config.py
5. Run main.py

## Configuration
1. Apart from standard argument based configuration used in previous work. additional configuration parameters has been provided in config/
2. Best hyperparameters for all the drugs including AUROC and AUPRC score hsa been provided in config/best_hypm.json with respect to the default seed in config/param_config.json
