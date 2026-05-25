from __future__ import annotations

import torch as pt
import matplotlib.pyplot as plt

from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable, Optional


def make_3d_grid( extent: float, 
                  n_grid: int,
                  *,
                  dtype: pt.dtype = pt.float64,
                  device: str | pt.device = "cpu",
                ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor, float]:
    """
    Create a cubic 3D grid.

    Arguments
    ---------
    extent : float
        Domain [-extend,extent] x [-extend,extent] x [-extend,extent]
    n_grid : int
        Number of grid points in each direction.
    dtype
        Default float64
    device
        Default CPU.

    Returns
    -------
    grid_points:
        Tensor of shape (n_grid**3, 3)
    X, Y, Z:
        Tensors of shape (n_grid, n_grid, n_grid)
    dx:
        Grid spacing.
    """

    xs = pt.linspace(-extent, extent, n_grid, dtype=dtype, device=device)
    ys = pt.linspace(-extent, extent, n_grid, dtype=dtype, device=device)
    zs = pt.linspace(-extent, extent, n_grid, dtype=dtype, device=device)

    dx = float(xs[1] - xs[0])

    X, Y, Z = pt.meshgrid(xs, ys, zs, indexing="ij")
    grid_points = pt.stack( (X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=1 )

    return grid_points, X, Y, Z, dx


def make_2d_slice_grid( extent: float,
                        n_grid: int,
                        *,
                        plane: str = "xz",
                        y_value: float = 0.0,
                        dtype: pt.dtype = pt.float64,
                        device: str | pt.device = "cpu",
                    ) -> tuple[pt.Tensor, pt.Tensor, pt.Tensor, float]:
    """
    Create a 2D slice grid through the molecule. Default plane is x-z at y=0, with x as bond axis.

    Arguments
    ---------
    extent : float
        Domain [-extend,extent] x [-extend,extent] x [-extend,extent]
    n_grid : int
        Number of grid points in each direction.

    Returns
    -------
    grid_points:
        Tensor of shape (n_grid**2, 3)
    A, B:
        Meshgrid coordinates for plotting.
    dx:
        Grid spacing.
    """

    a = pt.linspace(-extent, extent, n_grid, dtype=dtype, device=device)
    b = pt.linspace(-extent, extent, n_grid, dtype=dtype, device=device)

    dx = float(a[1] - a[0])

    A, B = pt.meshgrid(a, b, indexing="ij")

    if plane == "xz":
        X = A
        Y = pt.full_like(A, y_value)
        Z = B
    elif plane == "xy":
        X = A
        Y = B
        Z = pt.full_like(A, y_value)
    elif plane == "yz":
        X = pt.full_like(A, y_value)
        Y = A
        Z = B
    else:
        raise ValueError("plane must be one of 'xz', 'xy', or 'yz'.")

    grid_points = pt.stack( (X.reshape(-1), Y.reshape(-1), Z.reshape(-1)), dim=1 )
    return grid_points, A, B, dx


@pt.no_grad()
def estimate_one_electron_density_on_points(
    log_psi_fn: Callable[[pt.Tensor, pt.Tensor, pt.Tensor], pt.Tensor],
    R: pt.Tensor,
    grid_points: pt.Tensor,
    r2_samples: pt.Tensor,
    mc_weights: Optional[pt.Tensor] = None,
    *,
    batch_grid: int = 256,
    batch_r2: Optional[int] = None,
) -> pt.Tensor:
    """
    Estimate unnormalized one-electron density values on arbitrary points.

    We estimate

        density(r) ∝ ∫ |psi(r, r2)|^2 dr2

    using Monte Carlo samples r2_samples.

    Arguments
    ----------
    log_psi_fn:
        Function with signature

            u = log_psi_fn(R, r1, r2)

        where:
            R:  shape (B_R,)
            r1: shape (N, 3)
            r2: shape (N, 3)

        and output shape should be (B_R, N). For fixed R, B_R=1.

    R:
        Fixed nuclear geometry parameter, shape (1,) or compatible.

    grid_points:
        Points r where density is evaluated, shape (N_grid, 3).

    r2_samples:
        Monte Carlo samples for the second electron, shape (N_mc, 3).

    mc_weights:
        Optional importance weights proportional to 1/q(r2), shape (N_mc,).
        If None, uniform weights are used.

    batch_r2: int, optional
        Split large model evaluations into chunks of batch size `batch_r2`.

    Returns
    -------
    density:
        Unnormalized density values, shape (N_grid,).
    """

    device = grid_points.device
    dtype = grid_points.dtype

    R = R.flatten().to(device=device, dtype=dtype)
    r2_samples = r2_samples.to(device=device, dtype=dtype)

    n_grid = grid_points.shape[0]
    n_mc = r2_samples.shape[0]

    if mc_weights is None:
        mc_weights = pt.ones(n_mc, dtype=dtype, device=device)
    else:
        mc_weights = mc_weights.flatten().to(device=device, dtype=dtype)

    if mc_weights.shape[0] != n_mc:
        raise ValueError("mc_weights must have shape (N_mc,).")

    log_mc_weights = pt.log(mc_weights + 1e-8)

    density_chunks = []
    for g0 in range(0, n_grid, batch_grid):
        g1 = min(g0 + batch_grid, n_grid)
        points = grid_points[g0:g1]

        point_values = []

        for point in points:
            # Evaluate density at one r by integrating over r2.
            if batch_r2 is None:
                r1 = point[None, :].expand(n_mc, 3)
                r2 = r2_samples

                u = log_psi_fn(R, r1, r2).squeeze(0)  # (N_mc,)

                log_integrand = 2.0 * u + log_mc_weights
                log_integrand = log_integrand - log_integrand.max().detach()

                val = pt.exp(log_integrand).sum()
                point_values.append(val)

            else:
                vals_r2 = []

                for j0 in range(0, n_mc, batch_r2):
                    j1 = min(j0 + batch_r2, n_mc)

                    r2 = r2_samples[j0:j1]
                    r1 = point[None, :].expand(j1 - j0, 3)

                    u = log_psi_fn(R, r1, r2).squeeze(0)

                    log_integrand = 2.0 * u + log_mc_weights[j0:j1]
                    vals_r2.append(log_integrand)

                log_integrand = pt.cat(vals_r2, dim=0)
                log_integrand = log_integrand - log_integrand.max().detach()

                val = pt.exp(log_integrand).sum()
                point_values.append(val)

        density_chunks.append(pt.stack(point_values))

    return pt.cat( density_chunks, dim=0 )


def normalize_density_3d( density: pt.Tensor, dx: float ) -> pt.Tensor:
    """
    Normalize a 3D density array so that integral p(r) dr = 1.
    """

    dV = dx**3
    norm = density.sum() * dV
    return density / norm.clamp_min(1e-8)


def normalize_density_2d_for_plot( density_2d: pt.Tensor ) -> pt.Tensor:
    """
    Normalize 2D slice only for visualization.
    This is not a physical 3D normalization.
    """

    return density_2d / density_2d.max().clamp_min(1e-8)


def probability_isovalue( p_3d: pt.Tensor, dx: float, mass: float = 0.90 ) -> float:
    """
    Find density threshold c such that approximately

        ∫_{p(r) >= c} p(r) dr = mass.

    p_3d must be normalized as a 3D probability density.
    """

    dV = dx**3

    flat = p_3d.reshape(-1)
    sorted_vals, _ = pt.sort(flat, descending=True)

    cumulative = pt.cumsum(sorted_vals * dV, dim=0)

    target = pt.tensor(mass, dtype=p_3d.dtype, device=p_3d.device)
    idx = pt.searchsorted(cumulative, target)

    idx = pt.clamp(idx, 0, sorted_vals.numel() - 1)
    return float(sorted_vals[idx].detach().cpu())


def plot_density_slice_2d( density_2d: pt.Tensor,
                           A: pt.Tensor,
                           B: pt.Tensor,
                           *,
                           R_half: float,
                           plane: str = "xz",
                           title: str = "One-electron density slice",
                           levels: int = 40,
                        ) -> None:
    """
    Plot a 2D density slice.

    Assumes bond axis is x and nuclei are at x = ±R_half.
    """

    density_np = density_2d.detach().cpu().numpy()
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.contourf(A_np, B_np, density_np, levels=levels)
    plt.colorbar(label="normalized slice density")

    if plane == "xz":
        plt.scatter([-R_half, R_half], [0.0, 0.0], s=80, c="white", edgecolors="black")
        plt.text(-R_half, 0.12, "H", ha="center", va="bottom", color="white")
        plt.text(R_half, 0.12, "H", ha="center", va="bottom", color="white")
        plt.xlabel("x")
        plt.ylabel("z")
    elif plane == "xy":
        plt.scatter([-R_half, R_half], [0.0, 0.0], s=80, c="white", edgecolors="black")
        plt.xlabel("x")
        plt.ylabel("y")
    elif plane == "yz":
        plt.xlabel("y")
        plt.ylabel("z")

    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_density_isosurface_3d( p_3d: pt.Tensor,
                                dx: float,
                                extent: float,
                                *,
                                iso_value: float,
                                R_half: float,
                                title: str = "One-electron density isosurface",
                                alpha: float = 0.35,
                            ) -> None:
    """
    Plot a 3D density isosurface using marching cubes + matplotlib.

    p_3d:
        shape (nx, ny, nz), normalized probability density.
    dx:
        grid spacing.
    extent:
        grid runs from -extent to extent.
    iso_value:
        density threshold.
    """

    volume = p_3d.detach().cpu().numpy()
    verts, faces, normals, values = marching_cubes( volume, level=iso_value, spacing=(dx, dx, dx) )

    # marching_cubes coordinates start at 0; shift to physical coordinates.
    verts = verts - extent

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # nuclei
    ax.scatter([-R_half, R_half], [0, 0], [0, 0], s=120, c="black")
    ax.text(-R_half, 0, 0, "H", color="white", ha="center", va="center")
    ax.text(R_half, 0, 0, "H", color="white", ha="center", va="center")

    ax.plot([-R_half, R_half], [0, 0], [0, 0], c="black", linewidth=1)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()