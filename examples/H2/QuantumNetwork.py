import torch as pt
import torch.nn as nn

from typing import List

from qlp.MLP import MultiLayerPerceptron
from qlp.embedding import DistanceEmbedding

class QuantumNetwork( nn.Module ):
    """
    The main trainable neural network to solve the Schrodinger equation
    for the two-electron position-only wave function $\psi(r1, r2)$.
    """

    def __init__( self, 
                  neurons_per_layer : List[int],
                  r_cutoff : float,
                ) -> None:
        super().__init__()

        if neurons_per_layer[-1] != 1:
            raise ValueError("QuantumNetwork output dimension should be 1.")

        self.embedding = DistanceEmbedding()

        self.r_cutoff = r_cutoff
        self.eps = 1e-8
        self.mlp = MultiLayerPerceptron( neurons_per_layer, nn.GELU, init_zero=True )
        
        self.raw_alpha = nn.Parameter( pt.tensor(0.9) )
        self.raw_beta = nn.Parameter( pt.tensor(0.5) )

    def forward(self, R : pt.Tensor, # (B,)
                      r1 : pt.Tensor, # (N, 3)
                      r2 : pt.Tensor, # (N, 3)
                ) -> pt.Tensor:
        # Input Checks
        R = R.flatten()
        B = len( R )
        assert r1.ndim == 2 and r1.shape[1] == 3, f"`r1` must have shape (N, 3) but got {r1.shape}"
        N = r1.shape[0]
        assert r2.ndim == 2 and r2.shape[1] == 3, f"`r2` must have shape (N, 3) but got {r2.shape}"

        # Repmat, compute distances and put all in one big tensor
        P1 = pt.stack((-R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :] # (B, 1, 3)
        P2 = pt.stack(( R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :]
        r1 = r1[None,:,:].expand(B, N, 3)
        r2 = r2[None,:,:].expand(B, N, 3)
        d1A = pt.sqrt( pt.sum( (r1 - P1)**2, dim=2 ) + self.eps ) # (B,N)
        d1B = pt.sqrt( pt.sum( (r1 - P2)**2, dim=2 ) + self.eps)
        d2A = pt.sqrt( pt.sum( (r2 - P1)**2, dim=2 ) + self.eps)
        d2B = pt.sqrt( pt.sum( (r2 - P2)**2, dim=2 ) + self.eps)
        d12 = pt.sqrt( pt.sum( (r2 - r1)**2, dim=2 ) + self.eps)
        dAB = (2.0 * R)[:, None].expand(B, N)

        # Embed distances and pass through the network (B, N, 1+6*3)
        log_R = pt.log(R + self.eps)[:,None,None].expand(B, N, 1) 
        d1A_emb = self.embedding( d1A )
        d1B_emb = self.embedding( d1B )
        d2A_emb = self.embedding( d2A )
        d2B_emb = self.embedding( d2B )
        d12_emb = self.embedding( d12 )
        dAB_emb = self.embedding( dAB )
        mlp_input1 = pt.cat( (log_R, d1A_emb, d1B_emb, d2A_emb, d2B_emb, d12_emb, dAB_emb), dim=2 )
        mlp_output1 = self.mlp( mlp_input1 ) # shape (B, N, 1)
        mlp_output1 = pt.squeeze( mlp_output1, -1 )
        mlp_input2 = pt.cat( (log_R, d2A_emb, d2B_emb, d1A_emb, d1B_emb, d12_emb, dAB_emb), dim=2 )
        mlp_output2 = self.mlp( mlp_input2 ) # shape (B, N, 1)
        mlp_output2 = pt.squeeze( mlp_output2, -1 )

        # Assemble the log-wave function
        alpha = 2.0 * pt.sigmoid( self.raw_alpha )
        log_prefactor_1 = pt.logsumexp( -alpha*pt.stack( (d1A, d1B), dim=2), dim=2 )
        log_prefactor_2 = pt.logsumexp( -alpha*pt.stack( (d2A, d2B), dim=2), dim=2 )
        beta = 2.0 * pt.sigmoid( self.raw_beta )
        J = 0.5 * d12 / (1.0 + beta * d12 ) # Jastrow
        neural_sym = 0.5 * ( mlp_output1 + mlp_output2 )
        log_psi = log_prefactor_1 + log_prefactor_2 + J + neural_sym

        # Dirichlet gate.
        r1_sq = pt.sum( r1**2, dim=2 )
        dirichlet_gate1 = pt.relu( 1.0 - r1_sq / self.r_cutoff**2 )
        r2_sq = pt.sum( r2**2, dim=2 )
        dirichlet_gate2 = pt.relu( 1.0 - r2_sq / self.r_cutoff**2 )
        log_psi = pt.log( dirichlet_gate1 + self.eps ) + pt.log( dirichlet_gate2 + self.eps ) + log_psi

        return log_psi