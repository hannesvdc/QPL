import math
import torch as pt

from typing import Tuple

@pt.no_grad()
def sampleElectrons( N : int, gen : pt.Generator ) -> tuple[pt.Tensor, pt.Tensor]:
    # Sample (x,y,z) normal with a wider variance on the x-axis. 
    sigma_x = 2.0
    sigma_y = 1.0
    sigma_z = 1.0
    x = pt.normal( pt.zeros((N,)), sigma_x*pt.ones((N,)), generator=gen )
    y = pt.normal( pt.zeros((N,)), sigma_y*pt.ones((N,)), generator=gen )
    z = pt.normal( pt.zeros((N,)), sigma_z*pt.ones((N,)), generator=gen )
    xyz = pt.stack( (x,y,z), dim=1 )

    # symmetrize particles
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    neg_xyz = pt.stack( (-x,y,z), dim=1 )
    xyz = pt.cat( (xyz, neg_xyz), dim=0 )

    # Compute the MC weights
    exponent = -0.5 * ( (xyz[:,0] / sigma_x) ** 2 + (xyz[:,1] / sigma_y) ** 2 + (xyz[:,2] / sigma_z) ** 2 )
    q = pt.exp(exponent)  # proportional to q(x)
    mc_weights = 1.0 / q.clamp_min(1e-12)

    return xyz, mc_weights

@pt.no_grad()
def jointRejection( r1 : pt.Tensor, 
                    r2 : pt.Tensor, 
                    mc1 : pt.Tensor, 
                    mc2 : pt.Tensor,
                    R_cutoff : float,
                  ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    r1_sq = pt.sum( r1*r1, dim=1 )
    r2_sq = pt.sum( r2*r2, dim=1 )
    inside_domain = (r1_sq <= R_cutoff**2) & (r2_sq <= R_cutoff**2)

    r1 = r1[inside_domain,:]
    r2 = r2[inside_domain,:]
    mc_weights = mc1[inside_domain] * mc2[inside_domain]

    return r1, r2, mc_weights

@pt.no_grad()
def sampleBatch( B : int, 
                 N : int, 
                 R_cutoff : float, 
                 gen : pt.Generator 
                ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    log_R_min = math.log( 0.1 )
    log_R_max = math.log( 2.0 )
    log_R = log_R_min + (log_R_max - log_R_min) * pt.rand( (B,1), generator=gen )
    R = pt.exp( log_R )

    # Sample electrons
    r1, mc1 = sampleElectrons( N, gen )
    r2, mc2 = sampleElectrons( N, gen )
    r1, r2, mc_weights = jointRejection( r1, r2, mc1, mc2, R_cutoff )
    mc_weights /= mc_weights.mean()

    return R, r1, r2, mc_weights