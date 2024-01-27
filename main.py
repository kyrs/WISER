import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools
import numpy as np

from src.data import get_labeled_dataloader_generator
from config import data_config, param_config
from src import train_basis_code_adv
from src import basis_dataloader
from src.explainability_analysis import analyze_features
from src import fine_tuning
from copy import deepcopy
from src.cut_stats_subset import select_data

def generate_encoded_features(encoder, dataloader, normalize_flag=False):
    """

    :param normalize_flag:
    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0].cpu()
    label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor, label_tensor


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def main(args, update_params_dict):
    device = param_config.device
    cosine_flag = param_config.cosine_flag
    ccle_only = param_config.ccle_only   
    folder_name = param_config.folder_name
    seed = param_config.seed
    
    print(f"running experiment with CCLE only : {ccle_only} cosine flag : {cosine_flag}")
    eff_drug_list = param_config.eff_drug_list
    test_data_index  = param_config.test_data_index

    ############# data config ################
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)


    train_fn = train_basis_code_adv.train_code_adv
    with open(os.path.join('config/train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)

    if not args.norm_flag:
        method_save_folder = os.path.join(folder_name, args.method)
    else:
        method_save_folder = os.path.join(folder_name, f'{args.method}_norm')

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join(method_save_folder, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag,
            'testing_drug_len' : len(eff_drug_list),
        })
    
    safe_make_dir(training_params['model_save_folder'])

    
   
    ## loading data for alignment 
    s_dataloaders, t_dataloaders = basis_dataloader.get_dataloaders_for_alignment(
        drug_list = basis_drug_list,
        batch_size=training_params['unlabeled']['batch_size'],
        ccle_only= ccle_only, 
        seed = seed
    )

    # start Alignment  training
    ## NOTE: print and check inv_temp
    encoder, historys, basis_vec,  inv_temp = train_fn(s_dataloaders=s_dataloaders,
                                 t_dataloaders=t_dataloaders, ccle_only = ccle_only, 
                                 drug_dim = len(basis_drug_list), cosine_flag = cosine_flag, 
                                  **wrap_training_params(training_params, 
                                 type='unlabeled'))
    
    if args.retrain_flag:
        with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
                  'wb') as f:
            for history in historys:
                pickle.dump(dict(history), f)

    for drug in eff_drug_list:
        set_seed(param_config.seed)
        new_method_save_folder = os.path.join(folder_name, f'{args.method}_norm')
        task_save_folder = os.path.join(f'{new_method_save_folder}', args.measurement, drug)

        safe_make_dir(task_save_folder)

        ft_evaluation_metrics = defaultdict(list)
        labeled_dataloader_generator = get_labeled_dataloader_generator(
            gex_features_df=gex_features_df,
            seed=seed,
            batch_size=training_params['labeled']['batch_size'],
            drug=drug,
            ccle_measurement=args.measurement,
            threshold=args.a_thres,
            merged_tcga_add_data =None,
            days_threshold=args.days_thres,
            pdtc_flag=args.pdtc_flag,
            n_splits=args.n,
            return_unlabeled_tcga_flag = param_config.subset_selection_flag
            )


        fold_count = 0
        predicton_list_unlabeled = []
        for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader, unlabeled_tcga_dataloader, unlabeled_tcga_df in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            ft_basis_vec = deepcopy(basis_vec)
            
            target_classifier, info_unlabeled_tcga, ft_historys = fine_tuning.fine_tune_encoder_basis(
                encoder=ft_encoder,
                basis_vec = ft_basis_vec,
                inv_temp = inv_temp,
                cosine_flag=cosine_flag,
                train_dataloader=train_labeled_ccle_dataloader,
                val_dataloader=test_labeled_ccle_dataloader,
                test_dataloader=labeled_tcga_dataloader,
                seed=fold_count,
                unlabeled_tcga_dataloader = unlabeled_tcga_dataloader, 
                normalize_flag=args.norm_flag,
                metric_name=args.metric, 
                task_save_folder=task_save_folder,
                subset_selection_flag = param_config.subset_selection_flag,
                **wrap_training_params(training_params, type='labeled')
            )
            ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
            for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
                ft_evaluation_metrics[metric].append(ft_historys[test_data_index][metric][ft_historys[-2]['best_index']])
            fold_count += 1
        
            predicton_list_unlabeled.append(info_unlabeled_tcga)
        if args.hpt_flag == True:
            path_save = "alpha" + param_str.split("_alpha")[1]
            with open(os.path.join(task_save_folder, f'{path_save}_ft_evaluation_results.json'), 'w') as f:
                json.dump(ft_evaluation_metrics, f)
        else:
            with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
                json.dump(ft_evaluation_metrics, f)
        if param_config.subset_selection_flag:
            for budget in [0.1, 0.2, 0.4, 0.5 ,0.6, 0.8, 1]:
                set_seed(param_config.seed)
                index_to_select, associated_label = select_data(predicton_list_unlabeled, budget=budget)
                if len(index_to_select) == 0:
                    continue
                else:
                    ################## running again with new merged set##################
                    explain_stats = analyze_features(index_to_select = index_to_select, pseudo_labels=associated_label, fet_df=unlabeled_tcga_df, drugProcessed=drug)
                    fold_count=0
                    fet_tcga_train_select = torch.index_select(torch.from_numpy(unlabeled_tcga_df.values.astype('float32')),0, index_to_select)
                    labeled_ccle_merge_tcga_dataloader_generator = get_labeled_dataloader_generator(
                    gex_features_df=gex_features_df,
                    seed=seed,
                    batch_size=training_params['labeled']['batch_size'],
                    drug=drug,
                    merged_tcga_add_data = (fet_tcga_train_select,associated_label),
                    ccle_measurement=args.measurement,
                    threshold=args.a_thres,
                    days_threshold=args.days_thres,
                    pdtc_flag=args.pdtc_flag,
                    n_splits=args.n,
                    return_unlabeled_tcga_flag = False
                    )
                    ft_evaluation_metrics = defaultdict(list)
                    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader, unlabeled_tcga_dataloader, _ in labeled_ccle_merge_tcga_dataloader_generator:
                        target_classifier, ft_historys = fine_tuning.fine_tune_encoder_basis(
                            encoder=ft_encoder,
                            basis_vec = ft_basis_vec,
                            inv_temp = inv_temp,
                            cosine_flag=cosine_flag,
                            train_dataloader=train_labeled_ccle_dataloader,
                            val_dataloader=test_labeled_ccle_dataloader,
                            test_dataloader=labeled_tcga_dataloader,
                            seed=fold_count,
                            unlabeled_tcga_dataloader = unlabeled_tcga_dataloader, 
                            normalize_flag=args.norm_flag,
                            metric_name=args.metric, 
                            task_save_folder=task_save_folder,
                            subset_selection_flag = False,
                            **wrap_training_params(training_params, type='labeled')
                        )
                        ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
                        for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
                            ft_evaluation_metrics[metric].append(ft_historys[test_data_index][metric][ft_historys[-2]['best_index']])
                        fold_count += 1
                    
                    if args.hpt_flag == True:
                        path_save = "alpha" + param_str.split("_alpha")[1]
                        with open(os.path.join(task_save_folder, f'{path_save}_ft_evaluation_results_subset_{budget}.json'), 'w') as f:
                            json.dump(ft_evaluation_metrics, f)
                    else:
                        with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results_subset_{budget}.json'), 'w') as f:
                            json.dump(ft_evaluation_metrics, f)

                        if len(explain_stats) > 0:
                            with open(os.path.join(task_save_folder, f'explain_{param_str}_subset_{budget}_stats.json'), 'w') as f:
                                json.dump(explain_stats, f)
                        else:
                            pass
if __name__ == '__main__':
    
    basis_drug_list = param_config.basis_drug_list

    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv',
                        choices=['code_adv', 'dsn', 'dsna','code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae', 'ae'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])

    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)

    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--drug_dim', dest = 'drug_dim', nargs = '?', type = int, default = 7)
    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')

    parser.set_defaults(retrain_flag=True)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    # ***** added for hyperparameter tuning ****
    hpt_group = parser.add_mutually_exclusive_group(required=False)
    hpt_group.add_argument('--hpt', dest='hpt_flag', action='store_true')
    hpt_group.add_argument('--no-hpt', dest='hpt_flag', action='store_false')
    parser.set_defaults(hpt_flag=False)
    args = parser.parse_args()
    
    # params_grid = {
    # "pretrain_num_epochs": [50, 100, 300, 500, 800, 1000],
    # "train_num_epochs": [0, 1000, 1500, 2000, 2500, 3000],
    # "dop": [0.0, 0.1],
    # "inv_temp": [1, 1.5, 2, 5, 10, 50, 100, 1000]
    # }

    # params_grid = {
    # "pretrain_num_epochs": [50, 100, 300],
    # "train_num_epochs": [1000, 2000, 2500],
    # "dop": [0.1, 0.0],
    # "inv_temp": [0.01,0.1, 10, 2, 2.5, 1]
    # }

   
    # params_grid = {
    # "pretrain_num_epochs": [100,300,50],
    # "train_num_epochs": [1000,2500],
    # "train_num_epochs": [0.1,0.0],
    # "inv_temp": [0.25, 0.5, 2, 5, 100, 0.001]
    # }
    
    if args.method not in ['code_adv', 'adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    # # ************* FOR DOING RANDOMIZED GRIDSEARCH WITH DIFFERENT TEMPERATURE *******************
    # keys, values = zip(*params_grid.items())
    # update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # update_params_dict_list = random.sample(update_params_dict_list, 90) # to be removed later on (defaultl 80)
    # # # ***********************************************************************************

    # # # ************* FOR DOING GRIDSEARCH KEEPING TEMPERATURE CONSTANT!*******************
    # keys, values = zip(*params_grid.items())
    # update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # # ***********************************************************************************

    update_params_dict_list = [
                            #     Fu best hyperparameter
                                # {"pretrain_num_epochs":100, "train_num_epochs":1000, "dop":0.1, "inv_temp":100},
                                # {"pretrain_num_epochs":300, "train_num_epochs":1000, "dop":0.0, "inv_temp":100 }, 
                            #     GEM-best hyperparameter
                                # {"pretrain_num_epochs":300, "train_num_epochs":2000, "dop":0.1, "inv_temp":0.1},
                                # {"pretrain_num_epochs":100, "train_num_epochs":2500, "dop":0.0, "inv_temp":0.001 }

                            #     TEM-best hyperparameter
                                # {"pretrain_num_epochs":300, "train_num_epochs":3000, "dop":0.0, "inv_temp":100},
                            #     SOR-best hyperparameter
                                # {"pretrain_num_epochs":500, "train_num_epochs":2000, "dop":0.1, "inv_temp":2},
                            #     cis-best hyperparameter
                                {"pretrain_num_epochs":50, "train_num_epochs":2500, "dop":0.1, "inv_temp":2.5}
                               
                                ]
    # CHANGE FOLDER_NAME
    folder_name = 'model_save'

    for param_dict in update_params_dict_list:
        set_seed(param_config.seed)
        main(args=args, update_params_dict=param_dict)

