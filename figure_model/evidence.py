import torch
import torch.nn as nn
import math

# Original ccr_kernel, do not modify
# ccr_kernel = torch.tensor([
#     [
#         [1, 1],
#         [1, 0]
#     ],
#     [
#         [0, 0],
#         [0, 1]
#     ]
# ], dtype=torch.float32)

def evidence2prob_matrix(n: int) -> torch.Tensor:
    assert n >= 1 and isinstance(n, int), "n must be an integer >= 1"
    
    if n == 1:
        # Return a row vector of shape (1, 2): [0, 1]
        return torch.tensor([[0, 1]], dtype=torch.float32)
    
    # Recursively obtain a_k
    ak = evidence2prob_matrix(n - 1)
    
    # Construct row vectors of all zeros and all ones, matching the column count of a_k
    zeros = torch.zeros(1, ak.shape[1], dtype=ak.dtype)
    ones  = torch.ones (1, ak.shape[1], dtype=ak.dtype)
    
    # Top part [a_k, a_k]
    top = torch.cat([ak, ak], dim=1)
    # Bottom part [zeros..., ones...]
    bottom = torch.cat([zeros, ones], dim=1)
    
    # Combine to form a_{k+1}
    return torch.cat([top, bottom], dim=0)

def evidence2p_matrix(n: int) -> torch.Tensor:
    assert n >= 1 and isinstance(n, int), "n must be an integer >= 1"
    
    if n == 1:
        # Return a row vector of shape (1, 2): [0, 1]
        return torch.tensor([[0, 1]], dtype=torch.float32)
    
    # Recursively obtain a_k
    A = evidence2p_matrix(n - 1)
    
    # Construct row vectors of all zeros and all ones, matching the column count of a_k
    C = torch.zeros(1, A.shape[1], dtype=A.dtype)
    
    B = torch.ones(A.shape[0], A.shape[1], dtype=A.dtype)
    for ii in range(A.shape[0]):
        for jj in range(A.shape[1]):
            if A[ii, jj] == 0:
                B[ii, jj] = 0
            else:
                B[ii, jj] = 1 / ((1 / A[ii, jj]) + 1)
    
    B_sum = B.sum(dim=0, keepdim=True)

    D = torch.ones(1, A.shape[1], dtype=A.dtype) - B_sum

    # Top part [a_k, a_k]
    top = torch.cat([A, B], dim=1)
    # Bottom part [zeros..., ones...]
    bottom = torch.cat([C, D], dim=1)
    
    # Combine to form a_{k+1}
    return torch.cat([top, bottom], dim=0)[: , 1:]

class evidence2prob(nn.Module):
    def __init__(self, num_types: int):
        super().__init__()
        transform = evidence2prob_matrix(num_types)
        self.register_buffer('transform', transform)

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        # Convert evidence to probabilities
        prob = self.transform.matmul(mass.t()).t()
        row_sums = prob.sum(dim=1, keepdim=True)
        normalized = prob / row_sums
        return normalized                       



# class CCRServer(nn.Module):
#     def __init__(self, num_clients: int, device):
#         """
#         num_clients: number of clients to fuse
#         """
#         super().__init__()
#         self.num_clients = num_clients
#         # Register the kernel as a buffer so that model.to(device) moves it automatically to the GPU
#         self.register_buffer('kernel', ccr_kernel)
#         self.device = device

#     def forward(self, masses: torch.Tensor) -> torch.Tensor:
#         """
#         masses: Tensor of shape [B, num_clients, num_types]
#         returns: Tensor of shape [B, fused_dim]
#         """
#         # Take the first client's mass as the initial fused mass
#         fused = masses[:, 0, :]           # [B, num_types]
#         # Fuse sequentially with each of the remaining masses
#         n = int(math.log2(fused.shape[1]))
#         for i in range(1, self.num_clients):
#             fused = self._ccr(fused, masses[:, i, :], n)

#         # Convert evidence to probabilities
#         transform_tensor = evidence2prob(n).to(self.device)
#         final_prob = transform_tensor.matmul(fused.t()).t()
#         row_sums = final_prob.sum(dim=1, keepdim=True)
#         normalized = final_prob / row_sums
#         return normalized                       

#     def _kron_n_times(self, kernel: torch.Tensor, n: int) -> torch.Tensor:
#         """Compute the n-fold Kronecker product of the kernel"""
#         result = kernel
#         for _ in range(n - 1):
#             result = torch.kron(result, kernel)
#         return result

#     def _ccr(self,
#              mass1: torch.Tensor,
#              mass2: torch.Tensor,
#              n: int) -> torch.Tensor:
#         """
#         mass1, mass2: [B, num_types]
#         n: number of discernment frame elements, used to determine the order of Kronecker products
#         returns: [B, fused_dim]
#         """
#         # Build the combination tensor
#         combine_tensor = self._kron_n_times(self.kernel, n).to(self.device)  # [I, J, K]
#         # Perform batch fusion using einsum:
#         # out[b, i] = sum_{j,k} combine_tensor[i,j,k] * mass1[b,j] * mass2[b,k]
#         return torch.einsum('ijk,bj,bk->bi', combine_tensor, mass1, mass2)
