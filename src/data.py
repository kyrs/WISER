import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("../")
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader

from config import data_config,param_config
from src.data_preprocessing import align_feature

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as geo_dataLoader
import json 
from torch_geometric.utils import unbatch

def load_json(fileName):
    ## load relational data 
    listDict = []
    with open(fileName,"r+") as fileReader:
        for details in fileReader:
            dictVal = json.loads(details)
            listDict.append(dictVal)
    return listDict

def undirected_gene_coord_mat(listGeneDict, fetColList):
    ## generate the undirected coord matrix for graph level processing
    cntStatus = set()
    xCoord = []
    yCoord = []
    for val in listGeneDict:
        relationDict = val 
        relation = relationDict["relation"]
        geneX = relationDict["geneX"]
        geneY = relationDict["geneY"]
        out1  = list(fetColList).index(geneX)
        out2 =  list(fetColList).index(geneY)
        ind = (out1,out2)
        revInd = (out2, out1)   
        if ind not in cntStatus:
            cntStatus.add(ind)
            xCoord.append(out1)
            yCoord.append (out2)
        ## undirected graph , handling reverse cord
        if revInd not in cntStatus:
            cntStatus.add(revInd)
            xCoord.append(out2)
            yCoord.append(out1)

    coordIndex = np.vstack([xCoord, yCoord])
    return coordIndex

def directed_drug_fet_coord_mat(listDrugDict, fetColList):
    dictDrugMorgan = {}
    for dictVal in listDrugDict:
        if dictVal["drug"] not in dictDrugMorgan:
            dictDrugMorgan[dictVal["drug"]] = [{"gene": dictVal["gene"], "morgan_rep":dictVal["morgan_rep"]}]
        else:
            dictDrugMorgan[dictVal["drug"]].append({"gene": dictVal["gene"], "morgan_rep":dictVal["morgan_rep"]})
    

    drugList = sorted(dictDrugMorgan.keys()) ## sorting is essential to maintain order with labeled dataloader
    npFetList = []
    drug_gene_edge_x = []
    drug_gene_edge_y = []

    for drug in drugList:
        rep  = dictDrugMorgan[drug][0]["morgan_rep"]
        npFetList.append(rep)
        for index in range(len(dictDrugMorgan[drug])):
            gene = dictDrugMorgan[drug][index]["gene"]
            ## NOTE : CHECK each line of code
            geneId = list(fetColList).index(gene)
            drugId = drugList.index(drug)
            drug_gene_edge_x.append(drugId)
            drug_gene_edge_y.append(geneId)

    drugRep = np.vstack(npFetList)
    drug_gene_inter = np.vstack([drug_gene_edge_x, drug_gene_edge_y])  
    assert(drugRep.shape[0]==len(drugList)) ## ensure gene ordering is correct 
    assert(np.max(drug_gene_edge_x)==len(drugList)-1)
    return drugRep, drug_gene_inter, drugList


def graphMetaData(npFeature, npLabel, gene_gene_inter, drug_gene_inter, drug_fet , colFet):
    graphList = []
    assert(npFeature.shape[0] == npLabel.shape[0])
    for fet, label in zip(npFeature, npLabel):
        trainObj = HeteroData(gene={"x" : torch.tensor(fet).unsqueeze(1).type(torch.float), "num_nodes" :len(fet)},
                     drug={"x":torch.tensor(drug_fet).type(torch.float), "num_nodes" :len(drug_fet)}, 
                     label=torch.tensor(label).unsqueeze(0),
                     drug__inter__gene={"edge_index" :torch.tensor(drug_gene_inter)}, 
                     gene__inter__gene={"edge_index" : torch.tensor(gene_gene_inter)})
        graphList.append(trainObj)
    return graphList


def get_tcga_labeled_dataloaders(gex_features_df, drug, batch_size, days_threshold=None, tcga_cancer_type=None, graphLoader=True):
    if tcga_cancer_type is not None:
        raise NotImplementedError("Only support pan-cancer")

    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean() # >>> unlabeled gexfile

    tcga_treatment_df = pd.read_csv(data_config.tcga_first_treatment_file)
    tcga_response_df = pd.read_csv(data_config.tcga_first_response_file)

    tcga_treatment_df.drop_duplicates(subset=['bcr_patient_barcode'], keep=False, inplace=True) # >>> drug names
    tcga_treatment_df.set_index('bcr_patient_barcode', inplace=True)

    tcga_response_df.drop_duplicates(inplace=True) # >>> shows tumour relapse time
    tcga_response_df.drop_duplicates(subset=['bcr_patient_barcode'], inplace=True)
    tcga_response_df.set_index('bcr_patient_barcode', inplace=True)

    tcga_drug_barcodes = tcga_treatment_df.index[tcga_treatment_df['pharmaceutical_therapy_drug_name'] == drug]
    drug_tcga_response_df = tcga_response_df.loc[tcga_drug_barcodes.intersection(tcga_response_df.index)]
    labeled_tcga_gex_feature_df = tcga_gex_feature_df.loc[
        drug_tcga_response_df.index.intersection(tcga_gex_feature_df.index)]
    labeled_df = tcga_response_df.loc[labeled_tcga_gex_feature_df.index]
    # print(labeled_df)
    assert (all(labeled_df.index == labeled_tcga_gex_feature_df.index))

    # ********** added data for RECIST score **************************
    recist_data = pd.read_csv(data_config.tcga_drug_first_response_file)
    recist_data = recist_data.set_index('bcr_patient_barcode')
    # removing overlapping indices between relapse time and recist score dfs
    # recist_bcr = (recist_data.index.intersection(tcga_gex_feature_df.index.intersection(tcga_treatment_df.index))).difference(tcga_response_df.index.intersection(tcga_gex_feature_df.index.intersection(tcga_treatment_df.index)))

    recist_bcr = recist_data.index.intersection(tcga_gex_feature_df.index.intersection(tcga_treatment_df.index))

    recist_tcga_gex = tcga_gex_feature_df.loc[recist_bcr]
    recist_labels = recist_data.loc[recist_bcr]
    recist_drugs = tcga_treatment_df.loc[recist_bcr]

    thresholding = {'Stable Disease' : 0, 
                'Progressive Disease' : 0, 
                'Complete Response' : 1,
                'Complete Remission/Response' : 1, 
                'Partial Remission/Response' : 1,
                'Persistent Disease' : 0, 
                'Partial Response' : 1}

    recist_labels['treatment_outcome_at_tcga_followup'].replace(thresholding, inplace = True)

    recist_tcga_drug_barcodes = recist_drugs.index[recist_drugs['pharmaceutical_therapy_drug_name'] == drug]
    drug_tcga_recist_labels = recist_labels.loc[tcga_drug_barcodes.intersection(recist_labels.index)]
    recist_labeled_tcga_gex_feature_df = tcga_gex_feature_df.loc[
        drug_tcga_recist_labels.index.intersection(tcga_gex_feature_df.index)]
    recist_labeled_df = drug_tcga_recist_labels.loc[recist_labeled_tcga_gex_feature_df.index]
    recist_drug_labels = np.array(recist_labeled_df['treatment_outcome_at_tcga_followup'])

    assert (all(recist_labeled_df.index == recist_labeled_tcga_gex_feature_df.index))
    # ****************************************************************

    if days_threshold is None:
        days_threshold = np.median(labeled_df.days_to_new_tumor_event_after_initial_treatment)

    drug_label = np.array(labeled_df.days_to_new_tumor_event_after_initial_treatment > days_threshold, dtype='int32')
    #     drug_label = np.array(labeled_df.treatment_outcome_at_tcga_followup.apply(
    #     lambda s: s not in ['Progressive Disease', 'Stable Disease', 'Persistant Disease']), dtype='int32') #def implementation of RECIST values... (persistent...)

    # drug_label_df = pd.DataFrame(drug_label, index=labeled_df.index, columns=['label'])

    # ****************************** ADDED ******************************
    # new_labeled_tcga_gex_feature_df = pd.concat([labeled_tcga_gex_feature_df, recist_labeled_tcga_gex_feature_df], ignore_index = False)
    # new_drug_label = np.append(drug_label, recist_drug_labels)
    # *******************************************************************

    if not graphLoader:
        labeled_tcga_dateset = TensorDataset(
            torch.from_numpy(labeled_tcga_gex_feature_df.values.astype('float32')),
            torch.from_numpy(drug_label))

        labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                            batch_size=batch_size,
                                            shuffle=True)
    else:
        listGeneRel = load_json(data_config.gene_gene_relation)
        listDrugRel = load_json(data_config.drug_gene_relation)
        fetColList = gex_features_df.columns 
        # print(fetColList)
        gene_gene_inter = undirected_gene_coord_mat(listGeneDict = listGeneRel, fetColList = fetColList)
        drugFet, drug_gene_inter, drug_list_unsup = directed_drug_fet_coord_mat(listDrugDict = listDrugRel, fetColList = fetColList)
        assert(len(gex_features_df.columns)==1426) ## to ensure consistency in unsupervised and supervised dataloaders
        graph_node_list = graphMetaData(npFeature =labeled_tcga_gex_feature_df.values.astype('float32') , npLabel = drug_label, gene_gene_inter = gene_gene_inter, drug_gene_inter = drug_gene_inter, drug_fet = drugFet , colFet=gex_features_df.columns)
        labeled_tcga_dataloader = geo_dataLoader(graph_node_list, batch_size=batch_size, shuffle=True)
    return labeled_tcga_dataloader

def get_tcga_preprocessed_labeled_dataloaders(gex_features_df, drug, batch_size):
    if drug not in ['gem', 'fu']:
        raise NotImplementedError('Only support gem or fu!')
    non_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_non_gex.csv')
    res_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug}_res_gex.csv')

    non_feature_df = pd.read_csv(non_feature_file_path, index_col=0)
    _, non_feature_df = align_feature(gex_features_df, non_feature_df)

    res_feature_df = pd.read_csv(res_feature_file_path, index_col=0)
    _, res_feature_df = align_feature(gex_features_df, res_feature_df)

    raw_tcga_feature_df = pd.concat([non_feature_df, res_feature_df])

    tcga_label = np.ones(raw_tcga_feature_df.shape[0], dtype='int32')
    tcga_label[:len(non_feature_df)] = 0
    # tcga_label_df = pd.DataFrame(tcga_label, index=raw_tcga_feature_df.index, columns=['label'])

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(raw_tcga_feature_df.values.astype('float32')),
        torch.from_numpy(tcga_label))

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_tcga_dataloader


def get_pdtc_labeled_dataloaders(drug, batch_size, threshold=None, measurement='AUC'):
    pdtc_features_df = pd.read_csv(data_config.pdtc_gex_file, index_col=0)
    target_df = pd.read_csv(data_config.pdtc_target_file, index_col=0, sep='\t')
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, measurement]
    drug_feature_df = pdtc_features_df.loc[labeled_samples]

    assert all(drug_target_vec.index == drug_target_vec.index)

    if threshold is None:
        threshold = np.median(drug_target_vec)

    drug_label_vec = (drug_target_vec < threshold).astype('int')

    labeled_pdtc_dateset = TensorDataset(
        torch.from_numpy(drug_feature_df.values.astype('float32')),
        torch.from_numpy(drug_label_vec.values))

    labeled_pdtc_dataloader = DataLoader(labeled_pdtc_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_pdtc_dataloader


def get_ccle_labeled_dataloaders(gex_features_df, seed, drug, batch_size, ft_flag=False, threshold=None,
                                 measurement='AUC'):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    if ft_flag:
        train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels = train_test_split(
            ccle_labeled_feature_df.values,
            ccle_labels.values,
            test_size=0.1,
            stratify=ccle_labels.values,
            random_state=seed
        )

        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels))
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

    labeled_ccle_dateset = TensorDataset(
        torch.from_numpy(ccle_labeled_feature_df.values.astype('float32')),
        torch.from_numpy(ccle_labels.values))

    labeled_ccle_dataloader = DataLoader(labeled_ccle_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return (train_labeled_ccle_dataloader, test_labeled_ccle_dataloader) if ft_flag else labeled_ccle_dataloader


def get_ccle_labeled_dataloader_generator(gex_features_df, drug, batch_size, seed=2020, threshold=None,
                                          measurement='AUC', n_splits=5, graphLoader=True):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    s_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    if graphLoader:
        fetColList = gex_features_df.columns
        # print(fetColList)
        listGeneRel = load_json(data_config.gene_gene_relation)
        listDrugRel = load_json(data_config.drug_gene_relation)
        gene_gene_inter = undirected_gene_coord_mat(listGeneDict = listGeneRel, fetColList = fetColList)
        drugFet, drug_gene_inter, drug_list_unsup = directed_drug_fet_coord_mat(listDrugDict = listDrugRel, fetColList = fetColList)
    else:
        pass
    
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]

        
        if not graphLoader:
            train_labeled_ccle_dateset = TensorDataset(
                torch.from_numpy(train_labeled_ccle_df.astype('float32')),
                torch.from_numpy(train_ccle_labels))
            test_labeled_ccle_df = TensorDataset(
                torch.from_numpy(test_labeled_ccle_df.astype('float32')),
                torch.from_numpy(test_ccle_labels))

            train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

            test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                    batch_size=batch_size,
                                                  shuffle=True)
        else:
            
            assert(len(gex_features_df.columns)==1426) ## to ensure consistency in unsupervised and supervised dataloaders
            train_node_list = graphMetaData(npFeature = train_labeled_ccle_df.astype('float32') , npLabel = train_ccle_labels, gene_gene_inter = gene_gene_inter, drug_gene_inter = drug_gene_inter, drug_fet = drugFet , colFet=gex_features_df.columns)
            # print(train_node_list)
            train_labeled_ccle_dataloader = geo_dataLoader(train_node_list,batch_size=batch_size, shuffle=True)

            ### test data loader 
            test_node_list = graphMetaData(npFeature = test_labeled_ccle_df.astype('float32') , npLabel = test_ccle_labels, gene_gene_inter = gene_gene_inter, drug_gene_inter = drug_gene_inter, drug_fet = drugFet , colFet=gex_features_df.columns)
            test_labeled_ccle_dataloader = geo_dataLoader(test_node_list, batch_size=batch_size, shuffle=True)
            
            ## NOTE : double verify the ordering of train and test data
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


def get_labeled_dataloaders(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC', threshold=None,
                            days_threshold=None,
                            ft_flag=False,
                            pdtc_flag=False):
    """
    sensitive (responder): 1
    resistant (non-responder): 0
    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloaders = get_ccle_labeled_dataloaders(gex_features_df=gex_features_df,
                                                            threshold=threshold, seed=seed, drug=gdsc_drug,
                                                            batch_size=batch_size, ft_flag=ft_flag,
                                                            measurement=ccle_measurement)
    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold)

    return ccle_labeled_dataloaders, test_labeled_dataloaders


def get_labeled_dataloader_generator(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC', threshold=None,
                                     days_threshold=None,
                                     pdtc_flag=False,
                                     n_splits=5, graphLoader=True):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloader_generator = get_ccle_labeled_dataloader_generator(gex_features_df=gex_features_df,
                                                                              seed=seed,
                                                                              drug=gdsc_drug,
                                                                              batch_size=batch_size,
                                                                              threshold=threshold,
                                                                              measurement=ccle_measurement,
                                                                              n_splits=n_splits,
                                                                              graphLoader=graphLoader)

    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement,
                                                                graphLoader=graphLoader)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size, graphLoader=graphLoader)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold, graphLoader=graphLoader)

    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders


def get_ccle_labeled_tissue_dataloader_generator(gex_features_df, drug, batch_size, seed=2020, threshold=None,
                                                 measurement='AUC', n_splits=5, num_samples=12):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]

    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)

    ccle_sample_info.set_index('DepMap_ID', inplace=True)
    ccle_label_tissues = ccle_sample_info.loc[ccle_labeled_samples, 'lineage']
    assert all(ccle_labels.index == ccle_label_tissues.index)

    s_kfold = StratifiedKFold(n_splits=n_splits, random_state=seed)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]

        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(train_labeled_ccle_df.astype('float32')),
            torch.from_numpy(train_ccle_labels))
        test_labeled_ccle_df = TensorDataset(
            torch.from_numpy(test_labeled_ccle_df.astype('float32')),
            torch.from_numpy(test_ccle_labels))

        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_df,
                                                  batch_size=batch_size,
                                                  shuffle=True)

        train_samples = ccle_labeled_samples[train_index]
        train_sample_info = ccle_sample_info.loc[train_samples]
        train_tissue_counts = train_sample_info.lineage.value_counts()
        valid_tissues = train_tissue_counts.index[train_tissue_counts >= num_samples].tolist()
        tissue_dataloader_dict = dict()

        for tissue in valid_tissues:
            tissue_samples = train_sample_info.index[train_sample_info.lineage == tissue]
            tissue_feature_df = ccle_labeled_feature_df.loc[tissue_samples]
            tissue_label_df = ccle_labels.loc[tissue_samples]
            tissue_labeled_ccle_dateset = TensorDataset(
                torch.from_numpy(tissue_feature_df.values.astype('float32')),
                torch.from_numpy(tissue_label_df.values))

            tissue_labeled_ccle_dataloader = DataLoader(tissue_labeled_ccle_dateset,
                                                        batch_size=num_samples,
                                                        shuffle=True,
                                                        drop_last=True)
            tissue_dataloader_dict[tissue] = tissue_labeled_ccle_dataloader

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, tissue_dataloader_dict


def get_labeled_tissue_dataloader_generator(gex_features_df, drug, seed, batch_size, ccle_measurement='AUC',
                                            threshold=None,
                                            days_threshold=None,
                                            pdtc_flag=False,
                                            n_splits=5):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    if pdtc_flag:
        drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    else:
        drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    if drug in ['tgem', 'tfu']:
        gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']
    else:
        gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
        drug_name = drug_mapping_df.loc[drug, 'drug_name']

    print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')

    ccle_labeled_dataloader_generator = get_ccle_labeled_tissue_dataloader_generator(gex_features_df=gex_features_df,
                                                                                     seed=seed,
                                                                                     drug=gdsc_drug,
                                                                                     batch_size=batch_size,
                                                                                     threshold=threshold,
                                                                                     measurement=ccle_measurement,
                                                                                     n_splits=n_splits)

    if pdtc_flag:
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(drug=drug_name,
                                                                batch_size=batch_size,
                                                                threshold=threshold,
                                                                measurement=ccle_measurement)
    else:
        if drug in ['tgem', 'tfu']:
            test_labeled_dataloaders = get_tcga_preprocessed_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                                 drug=drug[1:],
                                                                                 batch_size=batch_size)
        else:
            test_labeled_dataloaders = get_tcga_labeled_dataloaders(gex_features_df=gex_features_df,
                                                                    drug=drug_name,
                                                                    batch_size=batch_size,
                                                                    days_threshold=days_threshold)

    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, tissue_dataloader_dict in ccle_labeled_dataloader_generator:
        return train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders, tissue_dataloader_dict


def get_adae_unlabeled_dataloaders(gex_features_df, batch_size, pos_gender='female'):
    sex_label_df = pd.read_csv(data_config.adae_sex_label_file, index_col=0, sep='\t')
    pos_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] == pos_gender])
    neg_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] != pos_gender])

    s_df = gex_features_df.loc[pos_samples]
    t_df = gex_features_df.loc[neg_samples]

    s_dataset = TensorDataset(
        torch.from_numpy(s_df.values.astype('float32'))
    )

    t_dataset = TensorDataset(
        torch.from_numpy(t_df.values.astype('float32'))
    )

    s_dataloader = DataLoader(s_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )

    t_dataloader = DataLoader(t_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )

    return (s_dataloader, s_dataloader), (t_dataloader, t_dataloader)


def get_adae_labeled_dataloaders(gex_features_df, seed, batch_size, pos_gender='female', ft_flag=False):
    """

    :param gex_features_df:
    :param seed:
    :param batch_size:
    :param pos_gender:
    :return:
    """
    sex_label_df = pd.read_csv(data_config.adae_sex_label_file, index_col=0, sep='\t')
    subtype_label_df = pd.read_csv(data_config.adae_subtype_label_file, index_col=0, sep='\t')
    gex_features_df = gex_features_df.loc[sex_label_df.index.intersection(gex_features_df.index)]
    subtype_label_df = subtype_label_df.loc[gex_features_df.index]
    assert all(gex_features_df.index == subtype_label_df.index)

    train_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] == pos_gender])
    test_samples = gex_features_df.index.intersection(sex_label_df.index[sex_label_df.iloc[:, 0] != pos_gender])

    train_df = gex_features_df.loc[train_samples]
    test_df = gex_features_df.loc[test_samples]

    if ft_flag:
        train_df, val_df, train_labels, val_labels = train_test_split(
            train_df.values,
            subtype_label_df.loc[train_samples].values,
            test_size=0.1,
            stratify=subtype_label_df.loc[train_samples].values,
            random_state=seed)

        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.astype('float32')),
            torch.from_numpy(train_labels.ravel())
        )

        val_labeled_dataset = TensorDataset(
            torch.from_numpy(val_df.astype('float32')),
            torch.from_numpy(val_labels.ravel())
        )

        val_labeled_dataloader = DataLoader(val_labeled_dataset,
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    else:
        train_labeled_dataset = TensorDataset(
            torch.from_numpy(train_df.values.astype('float32')),
            torch.from_numpy(subtype_label_df.loc[train_samples].values.ravel())
        )

    test_labeled_dataset = TensorDataset(
        torch.from_numpy(test_df.values.astype('float32')),
        torch.from_numpy(subtype_label_df.loc[test_samples].values.ravel())
    )

    train_labeled_dataloader = DataLoader(train_labeled_dataset,
                                          batch_size=batch_size,
                                          shuffle=True
                                          )

    test_labeled_dataloader = DataLoader(test_labeled_dataset,
                                         batch_size=batch_size,
                                         shuffle=True
                                         )

    return (train_labeled_dataloader, val_labeled_dataloader,
            test_labeled_dataloader) if ft_flag else (train_labeled_dataloader, test_labeled_dataloader)

if __name__ == "__main__":
    # print("Hello world !!")
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)
    labeled_dataloader_generator = get_labeled_dataloader_generator(
            gex_features_df=gex_features_df,
            seed=2023,
            batch_size=32,
            drug="fu",
            ccle_measurement="AUC",
            threshold=None,
            days_threshold=None,
            pdtc_flag=False,
            n_splits=5,
            graphLoader=True)
    
    for out in labeled_dataloader_generator:
        print(out)
        for batch in out[2]:
            print(batch)
            print(batch["gene"].batch)
            print(len(batch["gene"]["x"]), len(batch["gene"].batch), len(batch["label"]))
            print(unbatch(batch["gene"]["x"], batch["gene"].batch))
        # print(out)