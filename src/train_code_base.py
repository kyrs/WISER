import os
from itertools import chain

from src.dsn_ae import DSNAE
from src.evaluation_utils import *
from src.mlp import MLP
from torch_geometric.utils import unbatch



def eval_basis_dsnae_epoch(model, data_loader, device, history):
    """

    :param model:
    :param data_loader:
    :param device:
    :param history:
    :return:
    """
    model.eval()
    avg_loss_dict = defaultdict(float)
    
    graphLoader = True ## NOTE: need to change
    ##NOTE: doule check data loader
    for x_batch in data_loader:
        # print(f'x_batch : {x_batch[1].shape}')
        if not graphLoader:
            X_batch = x_batch[0].to(device)
            T_batch = x_batch[1].to(device)
        else:
            X_batch = x_batch.to(device)
            label = x_batch["label"]
            T_batch = label.to(device)

        with torch.no_grad():
            loss_dict = model.loss_function(X_batch, T_batch)
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history



def basis_dsn_ae_train_step(s_dsnae, t_dsnae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    s_dsnae.zero_grad()
    t_dsnae.zero_grad()
    s_dsnae.train()
    t_dsnae.train()

    graphLoader = True ## NOTE: need to change
    if not graphLoader:
        pass 
    else:
        s_x = s_batch.to(device)
        s_target = s_batch["label"].to(device)
        # s_label = unbatch(s_target, s_x["label"].batch)
        # s_target = torch.cat(s_label, dim =-1).to(device)

        t_x = t_batch.to(device)
        t_target = t_batch["label"].to(device)
        # t_label = unbatch(t_target, t_x["label"].batch)
        # t_target = torch.cat(t_label, dim=-1).to(device)
        ## NOTE: double check the label of source and target 
        
    # source_prediction = s_dsnae(s_x)
    # target_prediction = t_dsnae(t_x)
    
    s_loss_dict = s_dsnae.loss_function(s_x, s_target)
    t_loss_dict = t_dsnae.loss_function(t_x, t_target)

    optimizer.zero_grad()

    loss = s_loss_dict['loss'] + t_loss_dict['loss'] ## NOTE : hyperparameter not included
    loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)

    return history





