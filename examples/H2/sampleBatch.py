import math
import torch as pt

from typing import Tuple

@pt.no_grad()
def sampleElectrons( N : int, gen : pt.Generator, device : pt.device, dtype : pt.dtype ) -> tuple[pt.Tensor, pt.Tensor]:
    # Sample (x,y,z) normal with a wider variance on the x-axis. 
    sigma_x = 2.0
    sigma_y = 1.0
    sigma_z = 1.0
    mean = pt.zeros( (N,), device=device, dtype=dtype)
    stdev = pt.ones( (N,), device=device, dtype=dtype)
    x = pt.normal( mean, sigma_x*stdev, generator=gen )
    y = pt.normal( mean, sigma_y*stdev, generator=gen )
    z = pt.normal( mean, sigma_z*stdev, generator=gen )
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
def sampleSingleElectron( N : int, R_cutoff : float, gen : pt.Generator, device : pt.device, dtype : pt.dtype ) -> tuple[pt.Tensor, pt.Tensor]:
    xyz, mc_weights = sampleElectrons( N, gen, device, dtype )

    r_sq = pt.sum( xyz**2, dim=1 )
    inside_domain = (r_sq <= R_cutoff**2)

    xyz = xyz[inside_domain,:]
    mc_weights = mc_weights[inside_domain]

    return xyz, mc_weights / mc_weights.mean()

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
def sample_uniform_ball( N: int, 
                         R_cutoff: float,
                         *,
                         antithetic : bool = False,
                         gen: pt.Generator, 
                         device: pt.device, 
                         dtype: pt.dtype,
                        ) -> tuple[pt.Tensor, pt.Tensor]:
    # If antithetic=True, the returned number of samples is 8*N.
    x = pt.randn((N, 3), generator=gen, device="cpu", dtype=dtype )
    x = x / pt.linalg.norm(x, dim=1, keepdim=True).clamp_min(1e-300)
    u = pt.rand( (N, 1), generator=gen, device="cpu", dtype=dtype )

    r = R_cutoff * u.pow(1.0 / 3.0)
    pts = r * x

    if antithetic:
        # Reflections preserving the H2 geometry.
        sx = pt.tensor( [[ 1.0,  1.0,  1.0]], device="cpu", dtype=dtype) # (1,3)
        fx = pt.tensor( [[-1.0,  1.0,  1.0]], device="cpu", dtype=dtype)
        fy = pt.tensor( [[ 1.0, -1.0,  1.0]], device="cpu", dtype=dtype)
        fz = pt.tensor( [[ 1.0,  1.0, -1.0]], device="cpu", dtype=dtype)
        pts = pt.cat( ( pts * sx, pts * fx, pts * fy, pts * fz), dim=0 )

    weights = pt.ones((pts.shape[0],), device="cpu", dtype=dtype)

    return pts.to(device=device), weights.to(device=device)

@pt.no_grad()
def sampleBatch( B : int, 
                 N : int, 
                 R_cutoff : float, 
                 gen : pt.Generator,
                 device : pt.device,
                 dtype : pt.dtype,
                ) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    log_R_min = math.log( 0.1 )
    log_R_max = math.log( 2.0 )
    log_R = log_R_min + (log_R_max - log_R_min) * pt.rand( (B,1), generator=gen, device="cpu", dtype=dtype )
    R = pt.exp( log_R ).to( device=device )

    # Sample electrons
    r1, mc1 = sampleElectrons( N, gen, device, dtype )
    r2, mc2 = sampleElectrons( N, gen, device, dtype )
    r1, r2, mc_weights = jointRejection( r1, r2, mc1, mc2, R_cutoff )
    mc_weights /= mc_weights.mean()

    return R, r1, r2, mc_weights

@pt.no_grad()
def sampleBatchUniformBall( B: int, 
                            N: int, 
                            R_cutoff: float,
                            *,
                            antithetic : bool = False,
                            gen: pt.Generator,
                            device: pt.device,
                            dtype: pt.dtype,
                        ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    log_R_min = math.log(0.1)
    log_R_max = math.log(2.0)
    log_R = log_R_min + (log_R_max - log_R_min) * pt.rand( (B, 1), generator=gen, device="cpu", dtype=dtype )
    R = pt.exp(log_R).to(device=device)

    r1, _ = sample_uniform_ball(N, R_cutoff, antithetic=antithetic, gen=gen, device=device, dtype=dtype)
    r2, _ = sample_uniform_ball(N, R_cutoff, antithetic=antithetic, gen=gen, device=device, dtype=dtype)
    weights = pt.ones((r1.shape[0],), device=device, dtype=dtype)

    return R, r1, r2, weights