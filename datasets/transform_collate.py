from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Batch
from datasets.Data import NestedData, SubgData
from utils.atom_mass import atom_mass_dict


    
from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Batch
from datasets.Data import NestedData, SubgData
from utils.atom_mass import atom_mass_dict

import sys

class subg_transform(BaseTransform):
    def __init__(self, dname, subgraph_cutoff, cutoff, extend_r, max_neighbor, feature_dim=16, pos_enc_dim=16):
        self.dname = dname
        self.subgraph_cutoff = subgraph_cutoff
        self.cutoff = cutoff
        self.extend_r = extend_r
        self.max_neighbor = max_neighbor
        self.feature_dim = feature_dim
        self.pos_enc_dim = pos_enc_dim  # Added pos_enc_dim parameter
        
    def add_eig_vec(self, pos, edge_index):
        """
        Compute Laplacian eigenvector-based positional encodings
        """
        N = pos.size(0)
        device = pos.device
        
        # Compute full pairwise distance matrix
        #dist_matrix = torch.cdist(pos, pos)
        gram_matrix = pos @ pos.T
        # Create affinity matrix using Gaussian kernel
        #sigma = gram_matrix.mean()
        #A = torch.exp(-gram_matrix**2 / (2 * sigma**2))
        
        # Zero out non-edge connections
        #mask = torch.zeros((N, N), device=device, dtype=torch.bool)
        #mask[edge_index[0], edge_index[1]] = True
        #A = A * mask
        
        # Make symmetric
        #A = 0.5 * (A + A.T)
        
        # Compute normalized Laplacian
        #D = torch.diag(A.sum(dim=1))
        #D_inv_sqrt = torch.diag(torch.pow(A.sum(dim=1).clip(min=1e-6), -0.5))
        #L = torch.eye(N, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute eigenvectors
        eigval, eigvec = torch.linalg.eigh(gram_matrix)
        
        # Sort eigenvectors
        idx = torch.argsort(eigval, descending=True)
        eigval = eigval[idx]
        #print(eigval)
        eigvec = eigvec[:, idx]
        
        # Filter out near-zero eigenvalues
        tol = 1e-5
        mask = eigval > tol
        non_zero_eigval = eigval[mask]
        non_zero_eigvec = eigvec[:, mask]
        #print(non_zero_eigvec.size())
        # Take desired number of eigenvectors from non-zero ones
        pos_enc = non_zero_eigvec[:, :self.pos_enc_dim].float()
        # Get the available number of eigenvectors
        num_available = non_zero_eigvec.size(1)
        #print(pos_enc.size())
        # Pad if necessary
        if num_available < self.pos_enc_dim:
            pos_enc = torch.nn.functional.pad(pos_enc, (0, self.pos_enc_dim - num_available), value=0)
        #print(pos_enc.size()); print(pos_enc)
        return pos_enc

    def __call__(self, data):
        N = data.pos.shape[0]
        data.K = N

        # Compute edge_index for main graph first
        if self.cutoff is None:
            self.cutoff = torch.inf
        dist_matrix = (data.pos.unsqueeze(0) - data.pos.unsqueeze(1)).norm(dim=-1)
        global_mask = (dist_matrix <= self.cutoff) * (dist_matrix > 0.)
        edge_index = global_mask.nonzero(as_tuple=False).t()

        # Compute positional encoding for main graph
        pos_encoding = self.add_eig_vec(data.pos, edge_index)
        
        # Combine random features with positional encoding
        data.x = torch.cat([
            pos_encoding
        ], dim=-1)
        
        subg_datas = []
        for i in range(data.K):
            subg_data = self.subg_cal(
                data=data,
                subgraph_radius=self.extend_r,
                center_idx=i,
            )
            subg_datas.append(subg_data)
        
        loader = DataLoader(subg_datas, batch_size=len(subg_datas), shuffle=False)
        subg_datas_batched = next(iter(loader))

        nested_data = NestedData()
        for var in data.keys():
            nested_data[var] = data[var]
        for var in subg_datas_batched.keys():
            nested_data[var] = subg_datas_batched[var]
        nested_data["mass"] = torch.tensor([atom_mass_dict[zi.item()] for zi in data.z], dtype=torch.float32)
        nested_data["edge_index"] = edge_index
        nested_data.batch_index = torch.zeros(data.pos.shape[0], dtype=torch.long)
        
        return nested_data
        
    def subg_cal(self, data, subgraph_radius, center_idx):
        node_num = data.pos.shape[0]

        dist = (data.pos - data.pos[center_idx].view(1, -1)).norm(dim=1)
        dist_rank = torch.argsort(torch.argsort(dist))
        candidate_indices = dist_rank < self.max_neighbor
        ori_mask = dist <= subgraph_radius
        mask = ori_mask & candidate_indices

        subg_node_index = torch.arange(node_num, dtype=torch.long)[mask]
        subg_size_origin = ori_mask.sum()
        subg_size = mask.sum()
        
        subg_z = data.z[mask]
        
        # Get subgraph positions and compute edge index
        poss = data.pos[subg_node_index]
        distance_matrix = (poss.unsqueeze(0) - poss.unsqueeze(1)).norm(dim=-1)
        edge_candidates = (distance_matrix <= self.subgraph_cutoff) * (distance_matrix > 0.)
        subg_edge_index = edge_candidates.nonzero(as_tuple=False).t()
        
        # Compute positional encoding for subgraph
        #subg_pos_encoding = self.add_eig_vec(poss, subg_edge_index)
        
        # Combine random features with positional encoding for subgraph
        #subg_x = torch.cat([
        #    torch.randn(subg_size, self.feature_dim),
        #    subg_pos_encoding
        #], dim=-1)
        
        subg_x = data.x[mask]
        
        self_index = mask[:center_idx].sum()
        
        subg_node_label = torch.ones_like(subg_z)
        subg_node_label[self_index] = 2
        
        subg_node_center_index = torch.ones_like(subg_node_index) * center_idx
        subg_batch_index = torch.zeros(subg_size, dtype=torch.long)

        subg_data = SubgData()
        for var in ['subg_z', 'subg_node_label', 'subg_edge_index', 
                    'subg_node_index', 'subg_node_center_index', 
                    'subg_batch_index', 'subg_size', 'subg_x']: #added another variable subg_x
            subg_data[var] = locals()[var]
        
        return subg_data
def collate_(data_list, name=None):
    if name is not None:
        for i in range(len(data_list)):
            data_list[i].y = data_list[i].y[0, int(name)]
    
    data = Batch.from_data_list(data_list)
    
    return data

def transform_collate_(data_list, transform):
    new_data_list = []
    for data in data_list:
        new_data = transform(data)
        new_data_list.append(new_data)
    data = Batch.from_data_list(new_data_list)
    
    return data
