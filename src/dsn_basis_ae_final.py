import torch
from torch import nn
from torch.nn import functional as F
from src.base_ae import BaseAE
from src.mlp import MLP
from src.types_ import *
from typing import List                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
from torch_geometric.utils import unbatch
class DSNBasisAE(BaseAE):
    def __init__(self, shared_encoder, decoder, basis_vec,  num_geo_layer, input_dim: int, latent_dim: int, testing_drug_len : int, inv_temp : int = 1, alpha: float = 1.0, beta : float = 1.0, gamma : float = 1.0, eta : float = 1.0,
                 basis_weight = torch.ones(1), hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = False, norm_flag: bool = False, cns_basis_label_loss : bool = True, cosine_flag: bool = False,
                 **kwargs) -> None:

        super(DSNBasisAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.eta = eta
        self.basis_weight = basis_weight
        self.noise_flag = noise_flag
        self.dop = dop
        self.norm_flag = norm_flag
        self.inv_temp = inv_temp
        self.testing_drug_len = testing_drug_len

        self.cns_basis_label_loss = cns_basis_label_loss
        
        self.cosine_flag = cosine_flag
        self.num_geo_layer = num_geo_layer

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.basis_vec = basis_vec

    
        self.private_encoder =  MLP(input_dim=input_dim,
                                    output_dim=latent_dim,
                                    hidden_dims=hidden_dims,
                                    dop=self.dop)
        
            
            
        self.softmax = nn.Softmax(dim = -1)

    def p_encode(self, fetInput) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(fetInput + torch.randn_like(fetInput, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(fetInput)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, fetInput) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(fetInput + torch.randn_like(fetInput, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(fetInput)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def get_proj_val(self, fetInput) -> Tensor:
        p_latent_code = self.p_encode(fetInput)
        s_latent_code = self.s_encode(fetInput)
        
        weighted_basis = self.basis_vec.weight 
       
        if self.cosine_flag :
            s_latent_code = F.normalize(s_latent_code, p=2, dim=1)
            norm_weighted_basis = F.normalize( weighted_basis, p=2, dim=1)
            s_project_code = torch.matmul(s_latent_code, norm_weighted_basis.T)
            s_projected_val = self.softmax(s_project_code * self.inv_temp)
        else:
            s_project_code = torch.matmul(s_latent_code, weighted_basis.T)
            s_projected_val = self.softmax(s_project_code *self.inv_temp)

        return s_project_code, s_projected_val

    def encode(self, fetInput) -> Tensor:
        p_latent_code = self.p_encode(fetInput)
        s_latent_code = self.s_encode(fetInput)
        
        weighted_basis = self.basis_vec.weight 

       
        if self.cosine_flag :
            s_latent_code = F.normalize(s_latent_code, p=2, dim=1)
            norm_weighted_basis = F.normalize( weighted_basis, p=2, dim=1)
            s_project_code = torch.matmul(s_latent_code, norm_weighted_basis.T)
            s_projected_val = self.softmax(s_project_code * self.inv_temp)
        else:
            s_project_code = torch.matmul(s_latent_code, weighted_basis.T)
            s_projected_val = self.softmax(s_project_code *self.inv_temp)



        final_code = torch.matmul(s_projected_val, weighted_basis)
        return torch.cat((p_latent_code, final_code), dim=1), final_code, s_latent_code 
    
    
    def decode(self, z) -> Tensor:
        outputs = self.decoder(z)
        return outputs
    
    def forward(self, fetInput, **kwargs) -> List[Tensor]:
        z_concat, final_code, s_latent_code = self.encode(fetInput)
        return [fetInput, self.decode(z_concat), final_code, s_latent_code, z_concat]
    
    def loss_function(self, batch, target) -> dict:
        output = self.forward(batch)
        fetInput = output[0]
        recons = output[1]
        final_code = output[2]
        s_latent_code = output[3]
        z_concat = output[4]
        # loss_projection to be ignored when aligning CCLE and TCGA

        weighted_basis = self.basis_vec.weight 

        norm_basis_vec = torch.norm(weighted_basis, p = 2, dim = 1).detach()
        norm_basis_vec = norm_basis_vec.unsqueeze(1)
        norm_basis_vec = weighted_basis.div(norm_basis_vec.expand_as(weighted_basis) + 1e-6)

        norm_s_latent_code = torch.norm(s_latent_code.detach(), p = 2, dim = 1).detach()
        norm_s_latent_code = norm_s_latent_code.unsqueeze(1)
        norm_s_latent_code = s_latent_code.div(norm_s_latent_code.expand_as(s_latent_code) + 1e-6)
        
        
        
        if self.cns_basis_label_loss:
            pos_cosine_dis = (target==1) * (1-torch.matmul(norm_s_latent_code.detach(), norm_basis_vec.T))
            neg_cosine_dis = (target==0) * (1-torch.matmul(norm_s_latent_code.detach(), norm_basis_vec.T))
            posCos = pos_cosine_dis.sum() / ((target==1).sum())
            negCos = neg_cosine_dis.mean() /  ((target==0).sum())
            # print(posCos, negCos)
            basis_label_loss = torch.max( posCos-negCos+0.2, torch.tensor(0))

        
        assert(z_concat.shape[1] == 2 * self.latent_dim )    
        p_z = z_concat[:, :z_concat.shape[1] // 2]
        s_z = z_concat[:, z_concat.shape[1] // 2:]

        recons_loss = F.mse_loss(fetInput, recons)
        
        

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

        
        
        
            
        loss_commit = torch.mean((s_latent_code - final_code.detach())**2)
        loss_commit_reverse = torch.mean((s_latent_code.detach() - final_code)**2)

        if(torch.isnan(recons_loss)):
            print(f'basis_vec {self.basis_vec.weight}')
            assert(False)
        
        
        if(self.cns_basis_label_loss):
            loss = recons_loss + self.alpha * ortho_loss + self.beta * loss_commit + self.gamma * loss_commit_reverse + self.eta * basis_label_loss
            print(f'recons_loss : {recons_loss} ortho_loss : {ortho_loss} commit : {loss_commit} loss_commit_reverse : {loss_commit_reverse} loss_basis_label : {basis_label_loss} cns_basis : {self.cns_basis_label_loss}')
            return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss, "commit" : loss_commit, "loss_commit_reverse": loss_commit_reverse, "loss_basis_label" : basis_label_loss}
        else:
            loss = recons_loss + self.alpha * ortho_loss + self.beta * loss_commit + self.gamma * loss_commit_reverse
            print(f'recons_loss : {recons_loss} ortho_loss : {ortho_loss} commit : {loss_commit} loss_commit_reverse : {loss_commit_reverse} "loss_basis_label" : {torch.tensor(0)}')
            return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss, "commit" : loss_commit, "loss_commit_reverse": loss_commit_reverse, "loss_basis_label" : torch.tensor(0)}