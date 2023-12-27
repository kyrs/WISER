import numpy as np 
import torch 
from scipy.stats import mode

def LabelPred(prob, class_0_th, class_1_th):
    # print(prob)
    if prob <  class_0_th:
        return 0 
    if prob >  class_1_th:
        return 1
    else:
        return -1
def MajorityVote(row):
    ## return the max cnt of label ignoring -1
    ## else return -1
    row = np.array(row)
    if sum(row==-1) == len(row):
        return -1
    else:
        return mode(row[row!=-1]).mode[0]
def select_data(index_dict_list, class_0_th=0.3, class_1_th = 0.7):
    ## selecting the data points for downstream training
    fet_dict = {}
    label_dict = {}
    index_list = []
    fet_list_flag = False
    for index_dict in index_dict_list:
        for index in sorted(index_dict):
            if not fet_list_flag:
                fet_dict[index] = index_dict[index]["fet"]

            if index in label_dict:
                label_dict[index].append(LabelPred(index_dict[index]["prob"],class_0_th = class_0_th, class_1_th = class_1_th))
            else:
                label_dict[index] = [LabelPred(index_dict[index]["prob"],class_0_th = class_0_th, class_1_th = class_1_th)]
        fet_list_flag = False 

    fetArray = np.vstack([fet_dict[i] for i in range(len(label_dict))])
    labelArray = np.vstack([MajorityVote(label_dict[i]) for i in range(len(label_dict))])
    fetArray   = torch.tensor(fetArray)
    labelArray = torch.tensor(labelArray)

    non_abstrain = torch.tensor(labelArray!=-1).squeeze(1)
    abstrain = torch.tensor(labelArray==-1).squeeze(1)

    print("Non abstrain : ", sum(non_abstrain))
    print("abstrain : ", sum(abstrain))
    if sum(non_abstrain) > 10: ## atleast 10 samples
        non_abstrain_label = labelArray[non_abstrain].squeeze(1)
        non_abstain_fet = fetArray[non_abstrain]
        non_abstrain_index = torch.arange(len(labelArray))[non_abstrain]
        print(non_abstain_fet.shape,non_abstrain_label.shape)
        select_index = get_cutstat_inds(non_abstain_fet,non_abstrain_label)
        select_index = torch.tensor(select_index)

        final_label = torch.index_select(non_abstrain_label, 0, select_index)
        final_index = torch.index_select(non_abstrain_index, 0, select_index)
        print("final set  1: ", sum(final_label==1), "final set  0: ", sum(final_label==0))
        return final_index, final_label
    else:
        return [], []
def get_cutstat_inds(features, labels, coverage=0.5, K=20, device='cpu'):
        # move to CPU for memory issues on large dset
    pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

    N = labels.shape[0]
    dists_sorted = torch.argsort(pairwise_dists)
    neighbors = dists_sorted[:,:K]
    dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
    weights = 1/(1 + dists_nn)
    neighbors = neighbors.to(device)
    dists_nn = dists_nn.to(device)
    weights = weights.to(device)
    cut_vals = (labels[:,None] != labels[None,:]).long()
    cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
    # print(weights.shape, cut_neighbors.shape)
    Jp = (weights * cut_neighbors).sum(dim=1)
    weak_counts = torch.bincount(labels)
    weak_pct = weak_counts / weak_counts.sum()
    prior_probs = weak_pct[labels]
    mu_vals = (1-prior_probs) * weights.sum(dim=1)
    sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
    sigma_vals = torch.sqrt(sigma_vals)
    normalized = (Jp - mu_vals) / sigma_vals
    normalized = normalized.cpu()
    inds_sorted = torch.argsort(normalized)
    N_select = int(coverage * N)
    conf_inds = inds_sorted[:N_select]
    conf_inds = list(set(conf_inds.tolist()))
    return conf_inds