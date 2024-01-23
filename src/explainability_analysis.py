import json 
from scipy.stats import ttest_ind_from_stats, mannwhitneyu
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import sys
sys.path.append("../")
from config import data_config
with open(data_config.sig_gene_json, "r+") as fileReader:
    drug_gene_inter_dict = json.load(fileReader)

def analyze_features(index_to_select, pseudo_labels, fet_df, drugProcessed, selected_fet_flag = False, p_cutoff=0.05):

    if selected_fet_flag:
        fet_analyze_set = set()
        for drug in drug_gene_inter_dict:
            fet_analyze_set = fet_analyze_set | set(drug_gene_inter_dict[drug])

        fet_analyze_list = list(fet_analyze_set)
        selected_fet_df = fet_df[fet_analyze_list] 
    else:
        
        fet_analyze_list = fet_df.columns.tolist()
        selected_fet_df = fet_df[fet_analyze_list]
    selected_tensor = torch.index_select(torch.from_numpy(selected_fet_df.values.astype('float32')), 0, index_to_select)
    gene_test_details = {}
    print(f"pseudo labels size : {len(pseudo_labels)}")
    # print(f"pseudo labels : {len(pseudo_labels)}")
    assert(len(selected_tensor) == len(pseudo_labels))
    neg_tensor  = selected_tensor[pseudo_labels==0]
    pos_tensor  = selected_tensor[pseudo_labels==1]
    if len(neg_tensor)==0 or len(pos_tensor)==0:
        return {}
    else:
        pass
    
    assert(selected_tensor.shape[1] == len(fet_analyze_list))
    assert(selected_tensor.shape[1] == len(pos_tensor.mean(0)))
    assert(selected_tensor.shape[1] == len(pos_tensor.std(0)))
    assert(len(neg_tensor.std(0)) == len(pos_tensor.std(0)))
    assert(len(neg_tensor.mean(0)) == len(pos_tensor.mean(0)))
    # fet_ranking(feature_tensor = selected_tensor, pseudo_labels = pseudo_labels, featureList = fet_analyze_list, drugProcessed = drugProcessed)
    index_cnt = 0
    for gene, pos_mean, pos_std, neg_mean, neg_std  in zip(fet_analyze_list, pos_tensor.mean(0), pos_tensor.std(0), neg_tensor.mean(0), neg_tensor.std(0)):
        #scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
        # p_stats = ttest_ind_from_stats(pos_mean, pos_std, len(pos_tensor), neg_mean, neg_std, len(neg_tensor))
        p_stats = mannwhitneyu(pos_tensor[:,index_cnt], neg_tensor[:, index_cnt]) 
        index_cnt+=1
        if selected_fet_flag:
            print(f"gene : {gene}  pos_mean : {pos_mean} pos_std : {pos_std} sample pos : {len(pos_tensor)} neg_mean : {neg_mean} sample neg : {len(neg_tensor)} neg_std : {neg_std} p-val : {p_stats.pvalue}")
            gene_test_details[gene] = {"pos_mean" : pos_mean.item(), "pos_std" : pos_std.item(), "neg_mean" : neg_mean.item(), "neg_std": neg_std.item(), "pos_sample": len(pos_tensor), "neg_sample": len(neg_tensor), "stats" : p_stats.statistic, "p-val":p_stats.pvalue}
        else:
            
            if p_stats.pvalue < p_cutoff:
                # print(f"gene : {gene}  pos_mean : {pos_mean} pos_std : {pos_std} sample pos : {len(pos_tensor)} neg_mean : {neg_mean} sample neg : {len(neg_tensor)} neg_std : {neg_std} p-val : {p_stats.pvalue}")
                gene_test_details[gene] = {"pos_mean" : pos_mean.item(), "pos_std" : pos_std.item(), "neg_mean" : neg_mean.item(), "neg_std": neg_std.item(), "pos_sample": len(pos_tensor), "neg_sample": len(neg_tensor), "stats" : p_stats.statistic, "p-val":p_stats.pvalue}
            else:
                continue

    fet_ranking(feature_tensor = selected_tensor, pseudo_labels = pseudo_labels, featureList = fet_analyze_list, drugProcessed = drugProcessed, sigTestGene = gene_test_details)
    return gene_test_details

def fet_ranking(feature_tensor, pseudo_labels, featureList, drugProcessed, sigTestGene):

    drug_abbv = {"cis": "Cisplatin", "fu": "Fluorouracil", "gem": "Gemcitabine", "tem" : "Temozolomide"}
    clf = ExtraTreesClassifier()
    X = feature_tensor.detach().cpu().numpy()
    y = pseudo_labels.detach().cpu().numpy()
    clf    = clf.fit(X, y)
    lenFet = len(featureList)
    
    if drugProcessed in drug_abbv:
    #     sig_kg_gene_list = drug_gene_inter_dict[drug_abbv[drugProcessed]]
    #     # model  = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features = len(sig_kg_gene_list)) # details of selection https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
    #     selectFet = model.transform(np.array(featureList).reshape(1, lenFet))
    #     selectFet = np.squeeze(selectFet)
    #     common_gene = set(sig_kg_gene_list) & set(selectFet.ravel().tolist())
    #     missed_selected  = set(selectFet.ravel().tolist()) - set(sig_kg_gene_list)
    #     missed_signif = set(sig_kg_gene_list) - set(selectFet.ravel().tolist())
    #     print(f"drug : {drugProcessed} : common/ selected {(len(common_gene)/len(selectFet)*100)}%  selected/sig {(len(common_gene)/len(sig_kg_gene_list)*100)}% missed/selected  {(len(missed_selected)/len(selectFet)*100)}% ")
    # else:
    #     pass
        sig_kg_gene_dict = drug_gene_inter_dict[drug_abbv[drugProcessed]]
        model_mean  = SelectFromModel(clf, prefit=True, threshold="mean") # details of selection https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
        model_median  = SelectFromModel(clf, prefit=True, threshold="median")
        selectFet_mean = model_mean.transform(np.array(featureList).reshape(1, lenFet))
        selectFet_median = model_median.transform(np.array(featureList).reshape(1, lenFet))
        
        selectFet_mean = np.squeeze(selectFet_mean)
        selectFet_median = np.squeeze(selectFet_median)
        print(f"feature list : {len(featureList)}")
        assert(len(set(sig_kg_gene_dict["sig"])- set(featureList))==0)
        assert(len(set(sig_kg_gene_dict["un_sig"])- set(featureList))==0)
        
        TP_mean = set(sig_kg_gene_dict["sig"]) & set(selectFet_mean)
        TN_mean = set(sig_kg_gene_dict["un_sig"]) & (set(featureList) - set(selectFet_mean)) 
        FP_mean = set(selectFet_mean) & set(sig_kg_gene_dict["un_sig"])
        FN_mean = (set(featureList) - set(selectFet_mean)) & set(sig_kg_gene_dict["sig"])

        TP_median = set(sig_kg_gene_dict["sig"]) & set(selectFet_median)
        TN_median = set(sig_kg_gene_dict["un_sig"]) & (set(featureList) - set(selectFet_median))
        FP_median = set(selectFet_median) & set(sig_kg_gene_dict["un_sig"])
        FN_median = (set(featureList) - set(selectFet_median)) & set(sig_kg_gene_dict["sig"])

        print(f" drug : {drugProcessed} TP(mean) : {len(TP_mean)} TN(mean) : {len(TN_mean)} FP(mean) : {len(FP_mean)} FN(mean) : {len(FN_mean)} posGDIC : {len(set(sig_kg_gene_dict['sig']))}  negGDISC : {len(set(sig_kg_gene_dict['un_sig']))} posPred : {len(set(selectFet_mean))} negPred :{len(featureList)-len(set(selectFet_mean))}")
        print(f" drug : {drugProcessed} TP(median) : {len(TP_median)} TN(median) : {len(TN_median)} FP(median) : {len(FP_median)} FN(median) : {len(FN_median)} posGDIC : {len(set(sig_kg_gene_dict['sig']))}  negGDISC : {len(set(sig_kg_gene_dict['un_sig']))} posPred : {len(set(selectFet_median))} negPred :{len(featureList)-len(set(selectFet_median))}")
