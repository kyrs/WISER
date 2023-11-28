import torch
from torch import nn
from torch.nn import functional as F
from src.base_ae import BaseAE
from src.types_ import *
from typing import List

class DSNBasisAE(BaseAE):
    def __init__(self, shared_encoder, decoder, basis_vec, sparse_weight_vec, num_geo_layer, input_dim: int, latent_dim: int, testing_drug_len : int, inv_temp : int = 1, pseudo_conf_threshold = 0.9, alpha: float = 1.0, beta : float = 1.0, gamma : float = 1.0, eta : float = 1.0,
                 basis_weight = torch.ones(1), graphLoader=False,  psuedo_label_flag = False, psuedo_label_update_cnt = 50,  hidden_dims: List = None, dop: float = 0.1, noise_flag: bool = False, norm_flag: bool = False, cns_basis_label_loss : bool = True, cosine_flag: bool = False,
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

        self.psuedo_label_flag = psuedo_label_flag
        self.psuedo_label_update_cnt = psuedo_label_update_cnt
        self.pseudo_conf_threshold = pseudo_conf_threshold
        self.cns_basis_label_loss = cns_basis_label_loss
        
        self.cosine_flag = cosine_flag
        self.num_geo_layer = num_geo_layer

        self.graphLoader = graphLoader

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        self.basis_vec = basis_vec
        self.sparse_weight_vec = sparse_weight_vec
        
        # print('train_fn, dsnae : ', self.basis_vec.weight)
        # modules = []

    
        # modules.append(
        #     nn.Sequential(
        #         nn.Linear(input_dim, hidden_dims[0], bias=True),
        #         nn.ReLU(),
        #         nn.Dropout(self.dop)
        #     )
        # )

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),

        #             nn.ReLU(),
        #             nn.Dropout(self.dop)
        #         )
        #     )
        # modules.append(nn.Dropout(self.dop))
        # modules.append(nn.Linear(hidden_dims[-1], latent_dim, bias=True))

        if not self.graphLoader :    
            self.private_encoder =  MLP(input_dim=input_dim,
                                    output_dim=latent_dim,
                                    hidden_dims=hidden_dims,
                                    dop=self.dop).to(kwargs['device'])
        else:
            self.private_encoder = geo_MLP(
                                input_dim=input_dim,
                                output_dim=latent_dim,
                                hidden_dims=hidden_dims,
                                dop=dop,
                                num_geo_layer = num_geo_layer
                            ).to(kwargs['device'])
            
            
        self.softmax = nn.Softmax(dim = -1)

    def p_encode(self, input) -> Tensor:
        if self.noise_flag and self.training:
            if not self.graphLoader:
                latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
            else:
                input["gene"] = input["gene"] + torch.randn_like(input["gene"], requires_grad=False) * 0.1
                latent_code = self.private_encoder(input)
        else:
            latent_code = self.private_encoder(input)

        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(input)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def get_proj_val(self,  Tensor) -> Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)
        
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

    def encode(self, input: Tensor) -> Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)
        
        weighted_basis = self.basis_vec.weight 

       
        if self.cosine_flag :
            s_latent_code = F.normalize(s_latent_code, p=2, dim=1)
            norm_weighted_basis = F.normalize( weighted_basis, p=2, dim=1)
            s_project_code = torch.matmul(s_latent_code, norm_weighted_basis.T)
            s_projected_val = self.softmax(s_project_code * self.inv_temp)
        else:
            s_project_code = torch.matmul(s_latent_code, weighted_basis.T)
            s_projected_val = self.softmax(s_project_code *self.inv_temp)


        psuedo_labels = (s_projected_val >= self.pseudo_conf_threshold) * torch.ones_like(s_projected_val)
        psuedo_labels = psuedo_labels.long().detach()

        final_code = torch.matmul(s_projected_val, weighted_basis)
        return torch.cat((p_latent_code, final_code), dim=1), final_code, s_latent_code, psuedo_labels 
    
    
    def decode(self, z: Tensor) -> Tensor:
        outputs = self.decoder(z)
        return outputs
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z_concat, final_code, s_latent_code, psuedo_labels = self.encode(input)
        return [input, self.decode(z_concat), final_code, s_latent_code, z_concat, psuedo_labels]
    
    def loss_function(self, batch, target) -> dict:
        output = self.forward(batch)
        input = output[0]
        recons = output[1]
        final_code = output[2]
        s_latent_code = output[3]
        z_concat = output[4]
        psuedo_labels = output[5]
        # loss_projection to be ignored when aligning CCLE and TCGA

        weighted_basis = self.basis_vec.weight 

        norm_basis_vec = torch.norm(weighted_basis, p = 2, dim = 1).detach()
        norm_basis_vec = norm_basis_vec.unsqueeze(1)
        norm_basis_vec = weighted_basis.div(norm_basis_vec.expand_as(weighted_basis) + 1e-6)

        norm_s_latent_code = torch.norm(s_latent_code.detach(), p = 2, dim = 1).detach()
        norm_s_latent_code = norm_s_latent_code.unsqueeze(1)
        norm_s_latent_code = s_latent_code.div(norm_s_latent_code.expand_as(s_latent_code) + 1e-6)
        print(norm_s_latent_code.shape, norm_basis_vec.shape, target.shape)
        
        
        
        assert (self.psuedo_label_flag != self.cns_basis_label_loss) or ((self.psuedo_label_flag == False) and (self.cns_basis_label_loss == False))

        if self.cns_basis_label_loss:
            basis_label_loss = (target == 1) * torch.square(torch.matmul(norm_s_latent_code.detach(), norm_basis_vec.T) - 1)
            basis_label_loss = torch.sum(basis_label_loss, 1).mean(0) 

        if self.psuedo_label_flag:
            # print(psuedo_labels)
            basis_label_loss = (psuedo_labels == 1) * torch.square(torch.matmul(norm_s_latent_code.detach(), norm_basis_vec.T) - 1)
            basis_label_loss = torch.sum(basis_label_loss, 1).mean(0)


        
        
        assert(z_concat.shape[1] == 2 * self.latent_dim )    
        p_z = z_concat[:, :z_concat.shape[1] // 2]
        s_z = z_concat[:, z_concat.shape[1] // 2:]
        #NOTE : check the dimension of the  Z_concat shape    

        recons_loss = F.mse_loss(input, recons)
        # print(recons_loss)
        

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))

        ## NOTE : Check mean v.s cosine
        
        
            
        loss_commit = torch.mean((s_latent_code - final_code.detach())**2)
        loss_commit_reverse = torch.mean((s_latent_code.detach() - final_code)**2)

        # print(f'recons_loss : {recons_loss} ortho_loss : {ortho_loss} commit : {loss_commit} loss_commit_reverse : {loss_commit_reverse} loss_basis_label : {basis_label_loss}')
        if(torch.isnan(recons_loss)):
            print(f'weighted_basis {weighted_basis}')
            print(f'basis_vec {self.basis_vec.weight}')
            assert(False)
        
        print(f'alpha : {self.alpha} beta : {self.beta} gamma : {self.gamma} eta : {self.eta}')
        if(self.cns_basis_label_loss or self.psuedo_label_flag):
            loss = recons_loss + self.alpha * ortho_loss + self.beta * loss_commit + self.gamma * loss_commit_reverse + self.eta * basis_label_loss
            print(f'recons_loss : {recons_loss} ortho_loss : {ortho_loss} commit : {loss_commit} loss_commit_reverse : {loss_commit_reverse} loss_basis_label : {basis_label_loss}')
            return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss, "commit" : loss_commit, "loss_commit_reverse": loss_commit_reverse, "loss_basis_label" : basis_label_loss}
        else:
            loss = recons_loss + self.alpha * ortho_loss + self.beta * loss_commit + self.gamma * loss_commit_reverse
            print(f'recons_loss : {recons_loss} ortho_loss : {ortho_loss} commit : {loss_commit} loss_commit_reverse : {loss_commit_reverse} "loss_basis_label" : {torch.tensor(0)}')
            return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss, "commit" : loss_commit, "loss_commit_reverse": loss_commit_reverse, "loss_basis_label" : torch.tensor(0)}