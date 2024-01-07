import json 
from scipy.stats import ttest_ind_from_stats
import torch

def analyze_features(index_to_select, pseudo_labels, fet_df):
    drug_gene_inter_dict = {'Cisplatin': ['GSTM1', 'GSTP1', 'NQO1', 'MT2A', 'A2M', 'ABCC3'],
    'Fluorouracil': ['TYMS', 'TYMP', 'UPP1', 'TYMS', 'ABCC3'], 
    'Sorafenib': ['FGFR1'], 
    'Gemcitabine': ['TYMS']}
    fet_analyze_set = set()

    for drug in drug_gene_inter_dict:
        fet_analyze_set = fet_analyze_set | set(drug_gene_inter_dict[drug])

    fet_analyze_list = list(fet_analyze_set)
    selected_fet_df = fet_df[fet_analyze_list] 
    selected_tensor = torch.index_select(torch.from_numpy(fet_df.values.astype('float32')), 0, index_to_select)
    gene_ttest_details = {}

    neg_tensor  = selected_tensor[pseudo_labels==0]
    pos_tensor  = selected_tensor[pseudo_labels==1]
    if len(neg_tensor)==0 or len(pos_tensor)==0:
        return {}
    else:
        pass
    
    for gene, pos_mean, pos_std, neg_mean, neg_std  in zip(fet_analyze_list, pos_tensor.mean(0), pos_tensor.std(0), neg_tensor.mean(0), neg_tensor.std(0)):
        #scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
        p_stats = ttest_ind_from_stats(pos_mean, pos_std, len(pos_tensor), neg_mean, neg_std, len(neg_tensor))
        print(f"gene : {gene}  pos_mean : {pos_mean} pos_std : {pos_std} sample pos : {len(pos_tensor)} neg_mean : {neg_mean} sample neg : {len(neg_tensor)} neg_std : {neg_std} p-val : {p_stats.pvalue}")
        gene_ttest_details[gene] = {"pos_mean" : pos_mean.item(), "pos_std" : pos_std.item(), "neg_mean" : neg_mean.item(), "neg_std": neg_std.item(), "pos_sample": len(pos_tensor), "neg_sample": len(neg_tensor), "t-stats" : p_stats.statistic, "p-val":p_stats.pvalue}
    return gene_ttest_details