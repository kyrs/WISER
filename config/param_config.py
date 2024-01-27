import torch

results_data = {"TCGA" : -1, "CCLE" : -2}
cosine_flag = True
ccle_only = False 
############ Subset Selection #########
subset_selection_flag = True

seed = 2020
folder_name = "logs/explain_2"
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
eff_drug_list = ['cis', 'sor', 'tem', 'fu', 'gem']
test_data_index = results_data["TCGA"]
basis_drug_list = ['fu', 'tem', 'gem', 'cis', 'sor','sun', 'dox', 'tam', 'pac', 'car', 'Cetuximab', 'Methotrexate', 'Topotecan', 'Erlotinib', 'Irinotecan', 'Bicalutamide', 'Temsirolimus', 'Oxaliplatin', 'Docetaxel', 'Etoposide']