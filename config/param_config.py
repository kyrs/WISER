import torch

results_data = {"TCGA" : -1, "CCLE" : -2}

cosine_flag = True
ccle_only = False 
seed = 2020
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eff_drug_list = ['fu', 'tem', 'gem', 'cis', 'sor']
test_data_index = results_data["TCGA"]
basis_drug_list = ['fu', 'tem', 'gem', 'cis', 'sor','sun', 'dox', 'tam', 'pac', 'car', 'Cetuximab', 'Methotrexate', 'Topotecan', 'Erlotinib', 'Irinotecan', 'Bicalutamide', 'Temsirolimus', 'Oxaliplatin', 'Docetaxel', 'Etoposide']