from torch import nn
from src.types_ import *
from typing import List
from src.gradient_reversal import RevGrad
import torch 
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GraphConv
from torch_geometric.utils import unbatch

class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, dop: float = 0.1, act_fn=nn.SELU, out_fn=None, gr_flag=False, **kwargs) -> None:
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.dop = dop

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        if gr_flag:
            modules.append(RevGrad())

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.BatchNorm1d(hidden_dims[0]),
                act_fn(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.BatchNorm1d(hidden_dims[i + 1]),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )

        self.module = nn.Sequential(*modules)

        if out_fn is None:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim, bias=True),
                out_fn()
            )



    def forward(self, input: Tensor) -> Tensor:
        embed = self.module(input)
        output = self.output_layer(embed)

        return output

class geo_MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List = None, dop: float = 0.1, act_fn=nn.SELU, out_fn=None, num_geo_layer = None ,gr_flag=False, **kwargs) -> None:
            super(geo_MLP, self).__init__()
            self.output_dim = output_dim
            self.dop = dop

            if hidden_dims is None:
                hidden_dims = [32, 64, 128, 256, 512]

            modules = []
            if gr_flag:
                modules.append(RevGrad())

            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0], bias=True),
                    act_fn(),
                    nn.Dropout(self.dop)
                )
            )
            self.convs = torch.nn.ModuleList()
            for _ in range(num_geo_layer):
                conv = HeteroConv({
                    ('drug', 'inter', 'gene'): GATConv((-1, -1), 1, add_self_loops=False),
                    ('gene', 'inter', 'gene'): GATConv((-1, -1), 1),
                }, aggr='sum')
                self.convs.append(conv)
                ## NOTE : check working of conv 

            for i in range(len(hidden_dims) - 1):
                modules.append(
                    nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                        #nn.BatchNorm1d(hidden_dims[i + 1]),
                        act_fn(),
                        nn.Dropout(self.dop)
                    )
                )

            self.module = nn.Sequential(*modules)

            if out_fn is None:
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_dims[-1], output_dim, bias=True)
                )
            else:
                self.output_layer = nn.Sequential(
                    nn.Linear(hidden_dims[-1], output_dim, bias=True),
                    out_fn()
                )



    def forward(self, fetInput) :
        fet_dict  = fetInput.x_dict
        edge_dict = fetInput.edge_index_dict
        batch_idx = fetInput["gene"].batch 
        for conv in self.convs:
            fet_dict = conv(fet_dict, edge_dict)
            # print(fet_dict)
            ## NOTE : check if y is also getting modified
            fet_dict = {key: x.relu() for key, x in fet_dict.items()}
        


        outGraph = fet_dict['gene']
        x = unbatch(outGraph, batch_idx)
        out = torch.cat(x, dim =-1)
        embed = self.module(out.T)
        output = self.output_layer(embed)
        return output