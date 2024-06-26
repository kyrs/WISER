import torch.nn as nn
import torch
from src.types_ import *

class EncoderDecoder_basis(nn.Module):

    def __init__(self, encoder, decoder, basis_vec,  testing_drug_len, inv_temp = 1, normalize_flag=False, cosine_flag=True ):
        super(EncoderDecoder_basis, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalize_flag = normalize_flag
        self.basis_vec = basis_vec
        self.inv_temp = inv_temp
        self.testing_drug_len = testing_drug_len
        self.cosine_flag = cosine_flag
        print(f" fine tuning ==> cosine Flag : {self.cosine_flag} inv_temp : {self.inv_temp}, norm_flag : {self.normalize_flag}")
    def forward(self, inputFet) -> Tensor:
        
        if self.normalize_flag:
            ## NOTE : check norm flag
            encoded_input = self.encoder(inputFet)
            encoded_input =  nn.functional.normalize(encoded_input, p=2, dim=1)
        else:
            encoded_input = self.encoder(inputFet)

        weighted_basis = self.basis_vec.weight 


        if self.cosine_flag :
            encoded_input =  nn.functional.normalize(encoded_input, p=2, dim=1)
            norm_weighted_basis = nn.functional.normalize(weighted_basis, p=2, dim=1)
            project_code = torch.matmul(encoded_input, norm_weighted_basis.T.detach())
        else:
            project_code = torch.matmul(encoded_input, weighted_basis.T.detach())
 
        projected_val = nn.Softmax(dim = -1)(project_code * self.inv_temp)

        final_code = torch.matmul(projected_val, weighted_basis.detach())

        if self.normalize_flag:
            final_code = nn.functional.normalize(final_code, p=2, dim=1)
        output = self.decoder(final_code)
        return output, final_code

    def encode(self, input: Tensor) -> Tensor:
        
        if self.normalize_flag:
            encoded_input = self.encoder(input)
            encoded_input =  nn.functional.normalize(encoded_input, p=2, dim=1)
        else:
            encoded_input = self.encoder(input)


        if self.cosine_flag :
            encoded_input =  nn.functional.normalize(encoded_input, p=2, dim=1)
            print(weighted_basis.shape)
            norm_weighted_basis = nn.functional.normalize(weighted_basis, p=2, dim=1)
            print(norm_weighted_basis.shape)
            project_code = torch.matmul(encoded_input, norm_weighted_basis.T.detach())
        else:
            project_code = torch.matmul(encoded_input, weighted_basis.T.detach())

        projected_val = nn.Softmax(dim = 1)(project_code* self.inv_temp)
        final_code = torch.matmul(projected_val, self.basis_vec.weight.detach())
        return final_code
    

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
