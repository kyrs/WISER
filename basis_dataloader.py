import torch
from torch.utils.data import TensorDataset, DataLoader
import gzip
import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import data_config
from IPython.display import display
from sklearn.model_selection import train_test_split, StratifiedKFold
from data_preprocessing import align_feature
import warnings
warnings.filterwarnings("ignore")

# this function will return both unlabeled and labelled data for a drug in CCLE 

# argument of this function would be a string (drug name) and the GDSC files to prevent rep reading...


def prepare_CCLE_files():
    
#     reading files
    gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
    gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)
    
#     selecting relevant columns 
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', 'Z_SCORE']]
    
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()
    
#     mapping COSMIC ID to DepMap ID
    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')
    
    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
    
    CCLE_Files = {
        'gdsc1_sensitivity_df' : gdsc1_sensitivity_df,
        'gdsc2_sensitivity_df' : gdsc2_sensitivity_df,
        'gdsc_sample_mapping_dict' : gdsc_sample_mapping_dict,
        'gex_features_df' : gex_features_df
    }

    return CCLE_Files

def fetchCCLE_datafordrug(CCLE_Files, drug, diagnosis = False, threshold = None):    

    
    drug = drug.lower()
    
    gdsc1_sensitivity_df = CCLE_Files['gdsc1_sensitivity_df']
    gdsc2_sensitivity_df = CCLE_Files['gdsc2_sensitivity_df']
    gdsc_sample_mapping_dict = CCLE_Files['gdsc_sample_mapping_dict']
    gex_features_df = CCLE_Files['gex_features_df']
    
#     getting dfs corresponding to a drug
    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin([drug])] 
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin([drug])]
    
#     removing duplicate COSMIC IDs and taking mean of labels
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    
#     concat of 1 and 2
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    
#     mapping COSMIC ID to DepMap ID 
    target_df = gdsc_target_df.reset_index().pivot_table(values= gdsc_target_df.columns, index='COSMIC_ID', columns='DRUG_NAME')
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]
    
    ########### Additional Changes ####################
    #NOTE : Checks 
    # 1. class label for drug is same as what is there in original code for an 5 random drugs.
    # 2. Explanation   
    here = gdsc_target_df.columns.tolist()
    here.insert(0, 'COSMIC_ID')
    arr = np.concatenate((np.array(target_df.index).reshape(-1, 1), ), axis = 1)
    for metric in gdsc_target_df.columns:
        arr = np.concatenate((arr, target_df[metric][drug].values.reshape(-1, 1)), axis = 1)
    ccle_target_df = pd.DataFrame(arr, columns = here)
    ccle_target_df = ccle_target_df.set_index('COSMIC_ID')
    ccle_target_df.dropna(inplace = True)
    ########################################################################
#     DepMap IDs of those samples: for which gex and labels are present
#     so we have the gene expressions and all the labels for these samples in CCLE
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

#     thresholding on Z scores
    # NOTE : None v/s Zer0 : IMP
    if threshold == None:
        threshold = np.median(ccle_target_df['Z_SCORE'].loc[ccle_labeled_samples].values)

    # print(f"drug {drug}, threshold {threshold}")
#    generating labels
    ccle_labels = (ccle_target_df['Z_SCORE'].loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    ccle_target_df['tZ_SCORE'] = ccle_labels
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)
    
    unlabelled_data = ccle_labeled_feature_df
    labeled_data = ccle_target_df.loc[ccle_labeled_samples]
    labeled_data['drug_response'] = [{}] * len(labeled_data)
    if diagnosis == True:
        labeled_data['drug_response'] = labeled_data['tZ_SCORE'].apply(lambda x: {f"{drug} (Diagnosis)" : x})
    else:
        labeled_data['drug_response'] = labeled_data['tZ_SCORE'].apply(lambda x: {drug: x})
    # print(labeled_data)
    return unlabelled_data, labeled_data

def build_basis_CCLE(drug_list, CCLE_Files, seed):
    # CCLE_Files = prepare_CCLE_files()
    
    extra_drugs = ['Bicalutamide', 'Cetuximab', 'Docetaxel', 'Erlotinib', 'Etoposide', 'Irinotecan', 'Methotrexate', 'Oxaliplatin', 'Temsirolimus', 'Topotecan']
    # MAPPING DRUGS FOR CCLE to GDSC Names: 
    drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    drug_df_list = []
    for drug in drug_list:
        if drug in ['tgem', 'tfu']:
            gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
            # drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']... NOT USING drug_name (only used in pdtc and tcga)
        elif drug in extra_drugs:
            gdsc_drug = drug
        else:
            gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
            # drug_name = drug_mapping_df.loc[drug, 'drug_name']

        
        if drug in ['tgem', 'tfu']: 
            unlabelled_data, labeled_data = fetchCCLE_datafordrug(CCLE_Files, gdsc_drug, True)
        else:
            unlabelled_data, labeled_data = fetchCCLE_datafordrug(CCLE_Files, gdsc_drug)

        
        
        assert(len(labeled_data) > 0 ) # NOTE :  check with wrong name
        drug_df_list.append(labeled_data)

    # NOTE : check with original data loader (class label )
    assert(len(drug_df_list) > 0)

    combined_df = pd.concat(drug_df_list)
    unique_indices = combined_df.index.unique()
    common_data = defaultdict(dict)
    for index in unique_indices:
        dictionaries = combined_df.loc[index, 'drug_response']    
        for dictionary in dictionaries:
            if isinstance(dictionary, dict):
                for key, value in dictionary.items():
                    common_data[index][key] = value

    # Create the common DataFrame from the common data dictionary
    common_lbld_df = pd.DataFrame.from_dict(common_data, orient='index')
    common_lbld_df = common_lbld_df.fillna(-1)
    # NOTE : check 
    # Generate basis for each row
    basis_list = []
    for _, row in common_lbld_df.iterrows():
        basis = {}
        for col in common_lbld_df.columns:
            # check this ************ not working for a single drug ****************
            category = col.split('_')[0]  # Extract the category name 
            if row[col] == 1:
                basis[category] = 1
            elif row[col] == -1:
                basis[category] = -1
            else:
                basis[category] = 0
        basis_list.append(basis)
    # basis_list
    vectorized_list = [(list(d.values())) for d in basis_list]
    
#     assert statement needed?
    for d, l in zip(basis_list, vectorized_list):
        assert list(d.values()) == l, "Values order mismatch!"

    common_lbld_df['Basis'] = vectorized_list
    common_lbld_df = common_lbld_df[['Basis']] ## Choosing only basis data
    
    common_unlbld_df = pd.DataFrame()
    for drug in drug_list:
        if drug in ['tgem', 'tfu']:
            gdsc_drug = drug_mapping_df.loc[drug[1:], 'gdsc_name']
            # drug_name = drug_mapping_df.loc[drug[1:], 'drug_name']... NOT USING drug_name (only used in pdtc and tcga)
        elif drug in extra_drugs:
            gdsc_drug = drug
        else:
            gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
            # drug_name = drug_mapping_df.loc[drug, 'drug_name']

        unlbld, lbld = fetchCCLE_datafordrug(CCLE_Files, gdsc_drug)
        indices = common_lbld_df.index.intersection(unlbld.index)
        common_unlbld_df = common_unlbld_df.append(unlbld[unlbld.index.isin(indices)])
        common_unlbld_df = common_unlbld_df.drop_duplicates()
        
    
    train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels = train_test_split(
        common_unlbld_df.values,
        common_lbld_df.values,
        test_size=0.1,
        random_state = seed
        )
    return common_unlbld_df, (train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels)

def get_data_for_TCGA(drug_list, gex_features_df, days_threshold = None):
    drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    with gzip.open(data_config.xena_sample_file) as f:
        xena_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    xena_samples = xena_sample_info_df.index.intersection(gex_features_df.index)
    xena_sample_info_df = xena_sample_info_df.loc[xena_samples]
    xena_df = gex_features_df.loc[xena_samples]
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()
    tcga_treatment_df = pd.read_csv(data_config.tcga_first_treatment_file)
    tcga_response_df = pd.read_csv(data_config.tcga_first_response_file)

    tcga_treatment_df.drop_duplicates(subset=['bcr_patient_barcode'], keep=False, inplace=True)
    tcga_treatment_df.set_index('bcr_patient_barcode', inplace=True)

    tcga_response_df.drop_duplicates(inplace=True)
    tcga_response_df.drop_duplicates(subset=['bcr_patient_barcode'], inplace=True)
    tcga_response_df.set_index('bcr_patient_barcode', inplace=True)

    drug_df_list = []
    drug_df_list_diag = []
    total_tcga_drug_barcodes = set()
    df_unlabeled = pd.DataFrame()

    for drug in drug_list:

        # NOTE : not considering combination of drugs...
        if '+' in drug:
            raise Exception("Combination of drug is not handled in the codebase.")
            #NOTE : Check Exception

        if drug in ['tgem', 'tfu']:
            # ****************** handling Diagnosis Drugs ****************
            non_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug[1:]}_non_gex.csv')
            res_feature_file_path = os.path.join(data_config.preprocessed_data_folder, f'{drug[1:]}_res_gex.csv')

            non_feature_df = pd.read_csv(non_feature_file_path, index_col=0)
            _, non_feature_df = align_feature(gex_features_df, non_feature_df)

            res_feature_df = pd.read_csv(res_feature_file_path, index_col=0)
            _, res_feature_df = align_feature(gex_features_df, res_feature_df)

            raw_tcga_feature_df = pd.concat([non_feature_df, res_feature_df])
            raw_tcga_feature_df = raw_tcga_feature_df.reindex(columns = gex_features_df.columns)

            tcga_label = np.ones(raw_tcga_feature_df.shape[0], dtype='int32')
            tcga_label[:len(non_feature_df)] = 0

            labeled_df = pd.DataFrame({'index' : raw_tcga_feature_df.index, 'thresholded' : tcga_label})
            labeled_df = labeled_df.set_index('index')
            labeled_df['drug_response'] = [{}] * len(labeled_df)
            labeled_df['drug_response'] = labeled_df['thresholded'].apply(lambda x: {drug : x})
            df_unlabeled = df_unlabeled.append(raw_tcga_feature_df)
            drug_df_list.append(labeled_df)
            continue

        tcga_drug_barcodes = tcga_treatment_df.index[tcga_treatment_df['pharmaceutical_therapy_drug_name'] == drug_mapping_df.loc[drug, 'drug_name']]
        tcga_drug_barcodes = set(tcga_drug_barcodes) # take union here for

        drug_tcga_response_df = tcga_response_df.loc[tcga_drug_barcodes.intersection(tcga_response_df.index)]
        labeled_tcga_gex_feature_df = tcga_gex_feature_df.loc[drug_tcga_response_df.index.intersection(tcga_gex_feature_df.index)]
        labeled_df = tcga_response_df.loc[labeled_tcga_gex_feature_df.index]
        total_tcga_drug_barcodes = total_tcga_drug_barcodes.union(labeled_tcga_gex_feature_df.index)

        days_threshold = None
        if days_threshold is None:
            days_threshold = np.median(labeled_df.days_to_new_tumor_event_after_initial_treatment)

        tcga_labels = (labeled_df['days_to_new_tumor_event_after_initial_treatment'] > days_threshold).astype('int')
        labeled_df['thresholded'] = tcga_labels
        labeled_df = labeled_df[['thresholded']]
        labeled_df['drug_response'] = [{}] * len(labeled_df)
        labeled_df['drug_response'] = labeled_df['thresholded'].apply(lambda x: {drug_mapping_df.loc[drug, 'drug_name'] : x})
        drug_df_list.append(labeled_df)

    temp_drug_list = []
    for i in range(len(drug_list)):
        if drug_list[i] in ['tgem', 'tfu']:
            temp_drug_list.append(drug_list[i])
        else:
            drug_name = drug_mapping_df.loc[drug_list[i], 'drug_name']
            temp_drug_list.append(drug_name)

    # write that as diagnosis

    combined_df = pd.concat(drug_df_list)
    keys_to_modify = temp_drug_list
    df_labeled = combined_df['drug_response'].apply(lambda x: [x.get(key, -1) for key in keys_to_modify])
    for index in df_labeled.index:
        try:
            df_unlabeled = df_unlabeled.append(tcga_gex_feature_df.loc[index])
        except KeyError:
            pass

    return df_unlabeled, df_labeled

def get_unsupervised_data_TCGA(drug_list, gex_features_df):
    with gzip.open(data_config.xena_sample_file) as f:
        xena_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    xena_samples = xena_sample_info_df.index.intersection(gex_features_df.index)
    xena_sample_info_df = xena_sample_info_df.loc[xena_samples]
    xena_df = gex_features_df.loc[xena_samples]
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()
    df_unlabeled = tcga_gex_feature_df

    df_labeled = pd.DataFrame({'Sample' : df_unlabeled.index, 'drug_responses' : [[-1] * len(drug_list)] * len(df_unlabeled.index)})
    df_labeled = df_labeled.set_index('Sample')
    return df_unlabeled, df_labeled


def TCGA_DataLoaders(drug_list, batch_size, unlabeled_TCGA_data, labeled_TCGA_data, seed):
    train_labeled_tcga_df, test_labeled_tcga_df, train_tcga_labels, test_tcga_labels = train_test_split(
        unlabeled_TCGA_data.values,
        labeled_TCGA_data.values,
        test_size=0.1,
        random_state = seed
#     NOTE : not used stratification here as well
    )
    train_tcga_labels = np.squeeze(train_tcga_labels)
    tensor_train_tcga_labels = torch.stack([torch.tensor(lst) for lst in train_tcga_labels])

    test_tcga_labels = np.squeeze(test_tcga_labels)
    tensor_test_tcga_labels = torch.stack([torch.tensor(lst) for lst in test_tcga_labels])

    train_labeled_tcga_dataset = TensorDataset(
        torch.from_numpy(train_labeled_tcga_df.astype('float32')),
        tensor_train_tcga_labels
    )

    test_labeled_tcga_dataset = TensorDataset(
        torch.from_numpy(test_labeled_tcga_df.astype('float32')),
        tensor_test_tcga_labels
    )

    train_labeled_tcga_dataloader = DataLoader(
        train_labeled_tcga_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_labeled_tcga_dataloader = DataLoader(
        test_labeled_tcga_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_labeled_tcga_dataloader, test_labeled_tcga_dataloader

def get_dataloaders_for_alignment(drug_list, batch_size, ccle_only, seed):

    # *************** Getting Dataloaders for CCLE *****************
    # NOTE : hardcoded file reading...
    CCLE_Files = prepare_CCLE_files()
    unlabeled_CCLE_data, labeled_CCLE_data = build_basis_CCLE(drug_list, CCLE_Files, seed)
    print(len(drug_list))
    train_labeled_CCLE_dataloader, test_labeled_CCLE_dataloader = CCLE_DataLoaders(drug_list, batch_size, unlabeled_CCLE_data, labeled_CCLE_data, seed)
    
    
    # *************** Getting Dataloaders for TCGA ****************
    gex_features_df = CCLE_Files['gex_features_df']
    
    # unlabeled_TCGA_data, labeled_TCGA_data = get_data_for_TCGA(drug_list, gex_features_df) >>> used when aligning with supervised TCGA data
    unlabeled_TCGA_data, labeled_TCGA_data = get_unsupervised_data_TCGA(drug_list, gex_features_df) # >>> used when using just unsupervised TCGA data = gene expression of ~9k samples
    print(len(unlabeled_TCGA_data), len(labeled_TCGA_data))

    print(f"Warning : TCGA sample size is around :  {len(labeled_TCGA_data)}")
    #NOTE :  check V-IMP warning  
    train_labeled_TCGA_dataloader, test_labeled_TCGA_dataloader = TCGA_DataLoaders(drug_list, batch_size, unlabeled_TCGA_data, labeled_TCGA_data, seed)

    if ccle_only:
        return (train_labeled_CCLE_dataloader, test_labeled_CCLE_dataloader), (train_labeled_CCLE_dataloader, test_labeled_CCLE_dataloader) 
    else:
        # assert(not torch.eq(labeled_CCLE_data[0], labeled_TCGA_data[0]))
        return (train_labeled_CCLE_dataloader, test_labeled_CCLE_dataloader), (train_labeled_TCGA_dataloader, test_labeled_TCGA_dataloader)


# this function returns 2 DLs... labeled train and test of CCLE
def CCLE_DataLoaders(drug_list, batch_size, unlabeled_data, labeled_data, seed):
    
#     NOTE : in src code train test split was done by stratifying... here not stratifying...
    
#     ******************** UNLABELED DATASETS WITHOUT K-FOLD **********************
    train_unlabeled_ccle_df, test_unlabeled_ccle_df = train_test_split(unlabeled_data, test_size=0.1, random_state = seed)
    train_unlabeled_ccle_dataset = TensorDataset(torch.from_numpy(train_unlabeled_ccle_df.values.astype('float32')))
    test_unlabeled_ccle_dataset = TensorDataset(torch.from_numpy(test_unlabeled_ccle_df.values.astype('float32')))
    unlabeled_CCLE_dataset = TensorDataset(torch.from_numpy(unlabeled_data.values.astype('float32')))

#     ******************************************************************************
    
#     ******************** UNLABELED DATALOADERS WITHOUT K-FOLD **********************
     
    train_unlabeled_ccle_dataloader = DataLoader(train_unlabeled_ccle_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_unlabeled_ccle_dataloader = DataLoader(test_unlabeled_ccle_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_CCLE_dataloader = DataLoader(unlabeled_CCLE_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
#     ******************************************************************************
        
    
    
#     ******************** LABELED DATASETS WITHOUT K-FOLD **********************

    train_labeled_ccle_df, test_labeled_ccle_df, train_ccle_labels, test_ccle_labels = labeled_data
    
    train_ccle_labels = np.squeeze(train_ccle_labels)
    tensor_train_ccle_labels = torch.stack([torch.tensor(lst) for lst in train_ccle_labels])
    
    test_ccle_labels = np.squeeze(test_ccle_labels)
    tensor_test_ccle_labels = torch.stack([torch.tensor(lst) for lst in test_ccle_labels])
    
    # ccle_labels = np.squeeze(labeled_data)
    # tensor_ccle_labels =  torch.stack([torch.tensor(lst) for lst in ccle_labels])
    
    train_labeled_ccle_dataset = TensorDataset(
        torch.from_numpy(train_labeled_ccle_df.astype('float32')),
        tensor_train_ccle_labels)
    
    test_labeled_ccle_dataset = TensorDataset(
        torch.from_numpy(test_labeled_ccle_df.astype('float32')),
        tensor_test_ccle_labels)
    
    # labeled_ccle_dataset = TensorDataset(
    #     torch.from_numpy(unlabeled_data.values.astype('float32')),
    #     tensor_ccle_labels)
    
#     ******************************************************************************

#     ******************** LABELED DATALOADERS WITHOUT K-FOLD **********************

    train_labeled_ccle_dataloader = DataLoader(
        train_labeled_ccle_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True)
    
    test_labeled_ccle_dataloader = DataLoader(
        test_labeled_ccle_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last = True)
    
    # labeled_ccle_dataloader = DataLoader(
    #     labeled_ccle_dataset,
    #     batch_size=batch_size,
    #     shuffle=True)
    
#     ******************************************************************************
    
#     NOTE : for now passing train-test CCLE for both source and target
#     CCLE_DataLoaders[0]... for accessing unlabeled dataloaders(src and target)
#     CCLE_DataLoaders[1]... for accessing labeled dataloaders(src and target)... (train-test)

    return train_labeled_ccle_dataloader, test_labeled_ccle_dataloader
     
    
    