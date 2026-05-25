import os
from dotenv import load_dotenv
from pathlib import Path

import math
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from QuantumNetwork import QuantumNetwork
from EnergyLoss import EnergyLoss

from sampleBatch import sampleBatch, sampleSingleElectron

from isosurface import (make_2d_slice_grid, 
                        make_3d_grid, 
                        estimate_one_electron_density_on_points, 
                        normalize_density_2d_for_plot, 
                        normalize_density_3d, 
                        plot_density_slice_2d, 
                        plot_density_isosurface_3d, 
                        probability_isovalue)

load_dotenv()
store_directory = Path( os.getenv( "RESULTS_DIR", "./Results" ) ) / "H2"
device_str = os.getenv( "DEVICE", "cpu" )

# ============================================================
# Config
# ============================================================
r_cutoff = 5.0

# Wavefunction slice plot
R_slice = 1.0
x_min = -5.0
x_max = 5.0
n_x = 100000

# Energy curve
R_min = 0.1
R_max = 2.5
n_R = 100

def load_model( dtype : pt.dtype = pt.float32 ) -> QuantumNetwork:
    z = 64
    neurons_per_layer = [ 19, z, z, z, z, 1]
    model = QuantumNetwork( neurons_per_layer, r_cutoff )
    model.load_state_dict( pt.load( store_directory  / 'best_model.pth', weights_only=True ) )
    model.to( dtype=dtype )
    return model

def plot_training_convergence():
    train_data = np.load( store_directory / 'train_data.npy' )
    train_counter = train_data[:,0]
    train_losses = train_data[:,1]
    val_data = np.load( store_directory / 'validation_data.npy' )
    validation_counter = val_data[:,0]
    validation_losses = val_data[:,1]
    
    plt.plot( train_counter, train_losses, alpha=0.7, label='Training Loss')
    plt.plot( validation_counter, validation_losses, alpha=0.7, label=r'Validation Loss ($R = 1$)')
    plt.axhline(-1.1030, 0, np.max(train_counter), linestyle='--', alpha=0.7, label=r'Hartree Minimum Energy at $R=1$')
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'Loss' )
    plt.legend()
    plt.show()



# ---------------------------------------------------------------------
# Energy curve plotting
# ---------------------------------------------------------------------
def plot_energy_curve_with_loss(
    model,
    loss_fn,
    r1_val: pt.Tensor,
    r2_val: pt.Tensor,
    mc_weights_val: pt.Tensor,
    R_min: float = 0.1,
    R_max: float = 2.0,
    n_R: int = 100,
    logspace: bool = True, 
    display : bool = True ):
    """
    Plot electronic and total H2 energies versus R.

    Requires loss_fn to support:

        E_total = loss_fn( model, R_values, r1_val, r2_val mc_weights_val )

    Important:
        Do NOT wrap this in torch.no_grad(), because the loss needs
        autograd to compute grad_x psi for the kinetic term.
    """
    if logspace:
        R_values = pt.exp( pt.linspace( math.log(R_min), math.log(R_max), n_R, ) )
    else:
        R_values = pt.linspace(R_min, R_max, n_R)

    E_elec = []
    for idx in range(len(R_values)):
        R = R_values[idx]
        en_total = loss_fn( model, pt.tensor([R]), r1_val, r2_val, mc_weights_val, training=False )  # (n_R,)
        en = en_total - 1.0 / (2.0 * R)
        E_elec.append( float(en) )
    E_elec = pt.tensor( E_elec )
    E_total = E_elec + 1.0 / (2.0 * R_values)

    # Useful validation printout near R = 0.70055
    R_opt = 0.70055
    idx_R1 = int( pt.argmin(pt.abs(R_values - R_opt)).item() )
    print()
    print(f"Validation near R = {R_opt} a0")
    print("---------------------")
    print(f"R                    = {R_values[idx_R1].item():.6f}")
    print(f"E_elec               = {E_elec[idx_R1].item():.8f} Hartree")
    print(f"E_total              = {E_total[idx_R1].item():.8f} Hartree")

    plt.figure(figsize=(8, 5))
    plt.plot(R_values, E_elec, label="electronic energy")
    plt.plot(R_values, E_total, label="total energy")
    plt.axvline(1.0, linestyle="--", linewidth=1, label="R = 1")
    plt.axhline(-0.5, linestyle="--", linewidth=1, label=r"$E_{\text{total}} = -0.5$")
    plt.xlabel("R [Bohr], nuclei at (-R,0,0), (R,0,0)")
    plt.ylabel("Energy [Hartree]")
    plt.title("H₂⁺ energy curve")
    plt.legend()
    plt.tight_layout()
    if display:
        plt.show()

def main():
    dtype = pt.float64
    device = pt.device( 'cpu' )

    model = load_model( )
    loss_fn = EnergyLoss()

    # Plot the training convergence
    plot_training_convergence( )

    # Plot energy curve
    gen = pt.Generator()
    r_cutoff = 5.0
    _, r1_val, r2_val, mc_weights = sampleBatch( 1, n_x, r_cutoff, gen, device, dtype )
    plot_energy_curve_with_loss(
        model=model,
        loss_fn=loss_fn,
        r1_val=r1_val,
        r2_val=r2_val,
        mc_weights_val=mc_weights,
        R_min=R_min,
        R_max=R_max,
        n_R=n_R,
        logspace=True,
        display=False )
    
    # Make a plot of the one-electron density
    extent = 5.0
    R_half = 0.70055
    R = pt.tensor([R_half], dtype=dtype, device=device)

    # For density plotting, use ONE-electron samples and ONE-electron weights.
    r2_samples, r2_weights = sampleSingleElectron( N=50000, R_cutoff=r_cutoff, gen=gen, device=device, dtype=dtype )
    r2_samples = r2_samples.to(device=device, dtype=dtype)
    r2_weights = r2_weights.to(device=device, dtype=dtype)
    def log_psi_fn(R, r1, r2):
        return model(R, r1, r2)
    
    # 2D plot
    n_grid_2d = 150

    grid_2d, A, B, dx2 = make_2d_slice_grid( extent=extent, n_grid=n_grid_2d, plane="xz", y_value=0.0, dtype=dtype, device=device )
    density_2d_flat = estimate_one_electron_density_on_points( log_psi_fn=log_psi_fn, R=R, grid_points=grid_2d, r2_samples=r2_samples, mc_weights=r2_weights, batch_grid=256, batch_r2=5000 )
    density_2d = density_2d_flat.reshape(n_grid_2d, n_grid_2d)

    # This normalization is only for nice plotting of the slice.
    density_2d_plot = normalize_density_2d_for_plot(density_2d)
    plot_density_slice_2d( density_2d_plot, A, B, R_half=R_half, plane="xz", title="H2 one-electron density, x-z slice", display=False )

    # 3D plot
    n_grid_3d = 64
    grid_3d, X, Y, Z, dx = make_3d_grid( extent=extent, n_grid=n_grid_3d, dtype=dtype, device=device )
    density_3d_flat = estimate_one_electron_density_on_points( log_psi_fn=log_psi_fn, R=R, grid_points=grid_3d, r2_samples=r2_samples, mc_weights=r2_weights, batch_grid=64, batch_r2=5000 )
    density_3d = density_3d_flat.reshape(n_grid_3d, n_grid_3d, n_grid_3d)

    # Mask outside the domain where your wavefunction is meaningful.
    r_sq = X**2 + Y**2 + Z**2
    density_3d = pt.where( r_sq <= r_cutoff**2, density_3d, pt.zeros_like(density_3d) )

    # Normalize as a true 3D probability density p(r), integral p(r) dr = 1.
    p_3d = normalize_density_3d( density_3d, dx )
    iso_90 = probability_isovalue( p_3d, dx, mass=0.90 )
    print("90% probability isovalue:", iso_90 )
    plot_density_isosurface_3d( p_3d, dx, extent, iso_value=iso_90, R_half=R_half, title="H2 one-electron density: 90% probability isosurface", alpha=0.35, display=False )

    plt.show()

if __name__ == "__main__":
    main()