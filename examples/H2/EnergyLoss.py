import math
import torch as pt

class EnergyLoss ( pt.nn.Module ):
    """
    Rayleigh-energy loss for the Schrodinger equation

    Model signature:
        log psi = u = model( input )
    """
    def __init__( self, chunk_size : int = 8 ):
        super().__init__()
        self.chunk_size = chunk_size

    def forward( self, model : pt.nn.Module,
                       R : pt.Tensor, # (B,)
                       r1 : pt.Tensor, # (N,3)
                       r2 : pt.Tensor, # (N,3)
                       mc_weights : pt.Tensor, # (N,)
                       training : bool,
                ) -> float:
        # Input checks and formatting
        R = R.flatten()
        R = R.requires_grad_( False )
        P1 = pt.stack((-R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :] # (B, 1, 3)
        P2 = pt.stack(( R, pt.zeros_like(R), pt.zeros_like(R)), dim=-1)[:, None, :]

        assert r1.ndim == 2 and r1.shape[1] == 3, f"`r1` must have shape (N,3) but got {r1.shape}."
        N = r1.shape[0]
        assert r2.ndim == 2 and r2.shape[0] == N and r2.shape[1] == 3, f"`r2` must have shape (N,3) but got {r2.shape}."
        mc_weights = mc_weights.flatten()
        mc_weights.requires_grad_( False )
        assert len(mc_weights) == N

        # Evaluate per chunk
        B = len(R)
        n_chunks = int( math.ceil( B / self.chunk_size ) )
        total_loss = 0.0
        for chunk in range( n_chunks ):
            b = chunk * self.chunk_size
            e = min( (chunk+1) * self.chunk_size, len(R) )
            R_c = R[b:e]
            P1_c = P1[b:e,:,:]
            P2_c = P2[b:e,:,:]

            # Evaluate the network '$\log \psi = NN_theta(R_A, R_B, r1, r2)$
            u, du_dr1, du_dr2 = self.fcn_and_grads( model, R_c, r1, r2, training )

            # Compute all distances
            r1_ext = r1[None,:,:] # (1, N, 3)
            r2_ext = r2[None,:,:] # (1, N, 3)
            d1A = pt.sqrt( pt.sum( (r1_ext - P1_c)**2, dim=2 )) # (B,N)
            d1B = pt.sqrt( pt.sum( (r1_ext - P2_c)**2, dim=2 )) # (B,N)
            d2A = pt.sqrt( pt.sum( (r2_ext - P1_c)**2, dim=2 )) # (B,N)
            d2B = pt.sqrt( pt.sum( (r2_ext - P2_c)**2, dim=2 )) # (B,N)
            d12 = pt.sqrt( pt.sum( (r2_ext - r1_ext)**2, dim=2 )) # (1,N)
            dAB = pt.sqrt( pt.sum( (P2_c - P1_c)**2, dim=2 )) # (B, 1)

            # Adjust the MC weights by the actual wave function
            log_w = 2.0 * u
            log_w = log_w - log_w.max(dim=1, keepdim=True).values.detach()
            psi_weights = pt.exp( log_w ) # (B, N)
            unnormalized_weights = mc_weights[None,:] * psi_weights
            weights = unnormalized_weights / pt.sum( unnormalized_weights, dim=1, keepdim=True )

            # compute the energy
            momentum_r1 = 0.5 * pt.sum( du_dr1**2, dim=2 ) # (B, N)
            momentum_r2 = 0.5 * pt.sum( du_dr2**2, dim=2 ) # (B, N)
            V_term = -1.0 / d1A - 1.0 / d1B - 1.0 / d2A - 1.0 / d2B + 1.0 / d12 + 1.0 / dAB
            energy = pt.sum( (momentum_r1 + momentum_r2 + V_term) * weights, dim=1 ) # (B,)

            # Convert energy to loss and calculate its gradients
            chunk_loss = energy.sum() / B
            total_loss += float( energy.sum().detach().cpu() )
            if training:
                chunk_loss.backward()

        # Do some logging
        loss_avg = total_loss / B
        return loss_avg

    def fcn_and_grads( self,
                       model : pt.nn.Module,
                       R : pt.Tensor, # (B,)
                       r1 : pt.Tensor, # (N, 3)
                       r2 : pt.Tensor, # (N, 3)
                       training : bool,
                    ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Evaluates the PINN and its gradients with respect to electron positions.

        Arguments
        ---------
            model : The quantum PINN
            R : tensor, shape (B,) 
                Distance between each nucleus from the origin on the x-axis
            r1 : tensor, shape (N,3)
            r2 : tensor, shape (N,3)
        
        Returns
        -------
            u:      (B, N)
                Logarithm of the wave function.
            du_dr1: (B, N, 3)
                Derivative of u w.r.t. the first electron coordinates
            du_dr2: (B, N, 3)
                Derivative of u w.r.t. the second electron coordinates
        """

        # Make `r1` and `r2` leaf tensors.
        r1 = r1.detach().clone().requires_grad_(True)
        r2 = r2.detach().clone().requires_grad_(True)

        # Evaluate the model in tensorized form
        u = model( R, r1, r2 )
        
        # Compute gradients sequentially
        B = len( R )
        du_dr1 = []
        du_dr2 = []
        for b in range( B ):
            grad_r1_b = pt.autograd.grad( outputs=u[b].sum(), inputs=r1, 
                                         create_graph=training, retain_graph=True )[0]
            grad_r2_b = pt.autograd.grad( outputs=u[b].sum(), inputs=r2, 
                                         create_graph=training, retain_graph=True )[0]
            du_dr1.append( grad_r1_b )
            du_dr2.append( grad_r2_b )
        
        du_dr1 = pt.stack( du_dr1, dim=0 )
        du_dr2 = pt.stack( du_dr2, dim=0 )

        return u, du_dr1, du_dr2