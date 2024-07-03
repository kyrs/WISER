# WISER
[![made-with-python](https://img.shields.io/badge/Made%20with-Python3-1f425f.svg?color=purple)](https://www.python.org/)
## Cancer drug response prediction through Weak supervISion and supErvised Representation learning
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
3. Download benchmark datasets (CODE-AE) available at Zenodo [http://doi.org/10.5281/zenodo.4776448]   (version 2.0)
4. Changed the `root dir` in the `config/data_config.py` to the address where benchmark dataset is saved.
5. Run main.py

## Configuration
1. In addition to the standard argument-based configuration used in previous work, additional configuration parameters have been provided in `config/`.
