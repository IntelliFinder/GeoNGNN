import torch.nn as nn
from torch import Tensor
import torch
from layers.basis_layers import rbf_class_mapping
from layers.basic_layers import Envelope
import sys

class SparseMol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbound_upper: float,
                 max_z: int,
                 r_max: float = 10,
                 pos_enc_dim: int = 8,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=False,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        self.p = 6
        self.pos_enc_dim = pos_enc_dim
        self.envelope = Envelope(r_max=r_max, p=self.p)
        
    def add_eig_vec(self, pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Graph positional encoding using full distance matrix Laplacian eigenvectors
        
        Args:
            pos: Node positions tensor of shape (N, d) where N is number of nodes and d is dimension
            edge_index: Edge index tensor of shape (2, E) where E is number of edges
        Returns:
            Eigenvector matrix of shape (N, pos_enc_dim)
        """
        N = pos.size(0)
        device = pos.device
        
        # Compute full pairwise distance matrix
        dist_matrix = torch.cdist(pos, pos)
        
        # Create affinity matrix using Gaussian kernel
        sigma = dist_matrix.mean()  # Adaptive bandwidth
        A = torch.exp(-dist_matrix**2 / (2 * sigma**2))
        
        # Zero out non-edge connections using edge_index
        mask = torch.zeros((N, N), device=device, dtype=torch.bool)
        mask[edge_index[0], edge_index[1]] = True
        A = A * mask
        
        # Make symmetric
        A = 0.5 * (A + A.T)
        
        # Compute normalized Laplacian
        D = torch.diag(A.sum(dim=1))
        D_inv_sqrt = torch.diag(torch.pow(A.sum(dim=1).clip(min=1e-6), -0.5))
        L = torch.eye(N, device=device) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute eigenvectors
        eigval, eigvec = torch.linalg.eigh(L)
        
        # Sort eigenvectors by eigenvalues (ascending order)
        idx = torch.argsort(eigval)
        eigvec = eigvec[:, idx]
        eigval = eigval[idx]
        
        # Take the last pos_enc_dim eigenvectors (corresponding to largest eigenvalues)
        eigvec = eigvec[:, -self.pos_enc_dim:].float()
        #print(eigvec)
        # Pad if necessary (when N < pos_enc_dim)
        if N < self.pos_enc_dim:
            eigvec = torch.nn.functional.pad(eigvec, (0, self.pos_enc_dim - N), value=0)
            
        return eigvec
        
    def forward(self, z: Tensor, x: Tensor, pos: Tensor, edge_index: Tensor):
        '''
            z (N, ): atomic number
            pos (N, 3): atomic position
            edge_index (2, E): edge indices
        '''
        #TODO add Laplacian node features to the node features
        #print(x);print(x.size()); sys.exit(0)
        emb1 = self.z_emb(z)
        ev = pos[edge_index[0]] - pos[edge_index[1]] # (E, 3)
        el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
        ef = self.rbf_fn(el) # (E, ef_dim)
        
        smooth_coef = self.envelope(el)
        
        
        #caluculate edge feautres of projectors (assuming distinct eigenvalues)
        eigvec = self.add_eig_vec(pos, edge_index)[:,:3] # first 3 columns 
        projectors = torch.mul(eigvec[edge_index[0]], eigevec[edge_index[1]])
        return emb1, ef, smooth_coef#, eigvec




class Mol2Graph(nn.Module):
    def __init__(self,
                 z_hidden_dim: int,
                 ef_dim: int,
                 rbf: str,
                 rbf_trainable: bool,
                 rbound_upper: float,
                 max_z: int,
                 **kwargs):
        super().__init__()
        self.rbf_fn = rbf_class_mapping[rbf](
                    num_rbf=ef_dim, 
                    rbound_upper=rbound_upper, 
                    rbf_trainable=rbf_trainable,
                    **kwargs
                )
        self.z_emb = nn.Embedding(max_z + 1, z_hidden_dim, padding_idx=0)
        

    def forward(self, z: Tensor, pos: Tensor, **kwargs):
        '''
            z (B, N)
            pos (B, N, 3)
            emb1 (B, N, z_hidden_dim)
            ef (B, N, N, ef_dim)
            ev (B, N, N, 3)
        '''
        emb1 = self.z_emb(z)
        
        if kwargs.get("edge_index", None) is not None:
            edge_index = kwargs["edge_index"]
            ev = pos[edge_index[0]] - pos[edge_index[1]] # (E, 3)
            el = torch.norm(ev, dim=-1, keepdim=True) # (E, 1)
            ef = self.rbf_fn(el) # (E, ef_dim)
        else:
            B, N = z.shape[0], pos.shape[1]
            ev = pos.unsqueeze(2) - pos.unsqueeze(1)
            el = torch.norm(ev, dim=-1, keepdim=True)
            ef = self.rbf_fn(el.reshape(-1, 1)).reshape(B, N, N, -1)
        
        return emb1, ef

