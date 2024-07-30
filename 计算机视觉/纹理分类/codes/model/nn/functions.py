import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F


def aggregate(A, X, C):
    r""" Aggregate operation, aggregate the residuals of inputs (:math:`X`) with repect
    to the codewords (:math:`C`) with assignment weights (:math:`A`).

    .. math::

        e_{k} = \sum_{i=1}^{N} a_{ik} (x_i - d_k)

    Shape:
        - Input: :math:`A\in\mathcal{R}^{B\times N\times K}`
          :math:`X\in\mathcal{R}^{B\times N\times D}` :math:`C\in\mathcal{R}^{K\times D}`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times K\times D}`

    Examples:
        >>> B,N,K,D = 2,3,4,5
        >>> A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
        >>> X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
        >>> func = encoding.aggregate()
        >>> E = func(A, X, C)
    """
    # return _aggregate.apply(A, X, C)
    B,N,D = X.shape
    K,D = C.shape
    _X = torch.tile(X.view(B,N,1,D), (1,1,K,1)) # B N D -> B N K D
    _C = torch.tile(C.view(1,1,K,D), (B,N,1,1)) # K D   -> B N K D
    r = _X - _C # B N K D
    _A = torch.tile(A.view(B,N,K,1), (1,1,1,D)) # B N K -> B N K D
    E = torch.sum(torch.mul(_A, r), dim=1) # B N K D -> B K D
        
    return E # B K D

def scaled_l2(X, C, S):
    r""" scaled_l2 distance

    .. math::
        sl_{ik} = s_k \|x_i-c_k\|^2

    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    # return _scaled_l2.apply(X, C, S)
    B,N,D = X.shape
    K,D = C.shape
    _X = torch.tile(X.view(B,N,1,D), (1,1,K,1)) # B N D -> B N K D
    _C = torch.tile(C.view(1,1,K,D), (B,N,1,1)) # K D   -> B N K D
    L2 = torch.sqrt(torch.sum((_X - _C)**2, dim=-1)) # B N K D -> B N K
    _S = torch.tile(S.view(1,1,K), (B,N,1)) # K -> B N K
    SL = torch.mul(_S, L2)

    return SL

# Experimental
def pairwise_cosine(X, C, normalize=False):
    r"""Pairwise Cosine Similarity or Dot-product Similarity
    Shape:
        - Input: :math:`X\in\mathcal{R}^{B\times N\times D}`
          :math:`C\in\mathcal{R}^{K\times D}` :math:`S\in \mathcal{R}^K`
          (where :math:`B` is batch, :math:`N` is total number of features,
          :math:`K` is number is codewords, :math:`D` is feature dimensions.)
        - Output: :math:`E\in\mathcal{R}^{B\times N\times K}`
    """
    if normalize:
        X = F.normalize(X, dim=2, eps=1e-8)
        C = F.normalize(C, dim=1, eps=1e-8)
    return torch.matmul(X, C.t())

if __name__ == '__main__':
    B = 4
    N = 16
    D = 3
    X = Variable(torch.cuda.DoubleTensor(B,N,D).uniform_(-0.5,0.5), requires_grad=True)
    K = 10
    C = Variable(torch.cuda.DoubleTensor(K,D).uniform_(-0.5,0.5), requires_grad=True)
    S = Variable(torch.cuda.DoubleTensor(K).uniform_(-0.5,0.5), requires_grad=True)
    
    scaled_l2(X,C,S)    
    A = Variable(torch.cuda.DoubleTensor(B,N,K).uniform_(-0.5,0.5), requires_grad=True)
    aggregate(A,X,C)

    print(1)
