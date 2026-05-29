import os
from dotenv import load_dotenv
from pathlib import Path

import math
import numpy as np
import torch as pt
import torch.optim as optim
from qlp.utils import getGradientNorm

import matplotlib.pyplot as plt
import argparse

from QuantumNetwork import QuantumNetwork
from EnergyLoss import EnergyLoss
from sampleBatch import sampleBatchUniformBall

from typing import List

load_dotenv()
store_directory = Path( os.getenv( "RESULTS_DIR", "./Results" ) ) / "H2"
store_directory.mkdir( parents=True, exist_ok=True )
device_str = os.getenv( "DEVICE", "cpu" )
print( f'Training on {device_str}.')

parser = argparse.ArgumentParser()
parser.add_argument('--name', nargs='?', dest='name', required=True)
args = parser.parse_args()
name = args.name

# Do everything in double precision.
dtype = pt.float64
device = pt.device( device_str )
gen = pt.Generator( device='cpu' )

# Create a training and validation dataset
B = 256
N_train = 10_000
B_val = 1
N_validation = 100_000

# Setup the network
R_cutoff = 5.0
z = 64
neurons_per_layer = [ 1+6*3, z, z, z, z, 1]
model = QuantumNetwork( neurons_per_layer, R_cutoff )
model.to( device=device, dtype=dtype )
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Loss function.
chunk_size = 4
loss_fcn = EnergyLoss( chunk_size=chunk_size )

# Setup the optimizer and learning rate scheduler
lr = 1e-3
optimizer = optim.Adam( model.parameters(), lr, amsgrad=True )

# Scheduler: simple steps
step_size = 500
gamma = 0.1
n_steps = 5
scheduler = optim.lr_scheduler.StepLR( optimizer, step_size, gamma )

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []

# Sample a new dataset every time
def train_epoch( epoch : int ):
    model.train( )
    optimizer.zero_grad( set_to_none=True )

    # Sample new training points every epoch.
    R, r1, r2, mc_weights = sampleBatchUniformBall( B, N_train, R_cutoff, antithetic=True, gen=gen, device=device, dtype=dtype )
        
    # Compute the loss (backward is called per-chunk inside loss_fcn)
    loss = loss_fcn( model, R, r1, r2, mc_weights, training=True )
    loss_grad = getGradientNorm( model )

    # Update the weights internally
    pt.nn.utils.clip_grad_norm_( model.parameters(), max_norm=1.0 )
    optimizer.step( )

    # Keep track of important metrics
    train_counter.append( epoch-1)
    train_losses.append( loss )
    train_grads.append( float(loss_grad.detach().cpu().item()) )

    # Print some diagnostics
    print_str = (
        f"\nEpoch {epoch:04d} "
        f"Loss: {loss:.5e}  "
        f"Grad: {loss_grad.item():.3e}  "
        f"Lr: {optimizer.param_groups[0]['lr']:.3e}"
    )
    print(print_str)

# Validation function
val_R = pt.tensor( [0.70055], dtype=dtype, device=device )
validation_counter : List = []
validation_losses : List = []
def validate_epoch( epoch : int ) -> float:
    # Resample validation electrons and weights to reduce effect of Monte Carlo noise.
    _, val_r1, val_r2, val_mc_weights = sampleBatchUniformBall( B_val, N_validation, R_cutoff, antithetic=True, gen=gen, device=device, dtype=dtype )

    # Compute the loss (no gradients needed for validation)
    total_energy = loss_fcn( model, val_R, val_r1, val_r2, val_mc_weights, training=False )

    # Log some interesting info
    proton_energy = 1.0 / (2.0 * float(val_R.item()) )
    electron_energy = total_energy - proton_energy

    # Store
    validation_counter.append( epoch )
    validation_losses.append( float(total_energy) )

    # Print and done.
    print_str = f'\nValidation Epoch {epoch:03d}: \tElectron Energy: {electron_energy:.5e} \tTotal Energy {total_energy:.5e}'
    print(print_str)

    return total_energy

# Main training loop
n_epochs = step_size * n_steps
best_val_loss = math.inf
print( validate_epoch( 0 ) )
try:
    for epoch in range( 1, n_epochs+1 ):
        # Train using the new dataset
        train_epoch( epoch )

        # Validate on independent but fixed data
        val_loss = validate_epoch( epoch )

        scheduler.step( )

        # Store the current model and optimizer weights.
        pt.save( model.state_dict(), store_directory / f"{name}_model_adam.pth")
        pt.save( optimizer.state_dict(), store_directory / f"{name}_optimizer_adam.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Storing the best model.")
            pt.save( model.state_dict(), store_directory / f"{name}_best_model.pth")
except KeyboardInterrupt:
    print( 'Aborting Training.')
    pass

# Make numpy arrays from the training data
train_counter = np.array( train_counter ) # type: ignore
train_losses = np.array( train_losses ) # type: ignore
train_grads = np.array( train_grads ) # type: ignore
train_data = np.stack( (train_counter, train_losses, train_grads), axis=1) # type: ignore
validation_counter = np.array( validation_counter ) # type: ignore
validation_losses = np.array( validation_losses ) # type: ignore
validation_data = np.stack( (validation_counter, validation_losses), axis=1) # type: ignore
np.save( store_directory / f"{name}_train_data.npy", train_data)
np.save( store_directory / f"{name}_validation_data.npy", validation_data)

# Make a plot of the training progress
fig = plt.figure()
ax1 = fig.gca()
ax1.plot(train_counter, train_losses, color="tab:blue", alpha=0.7, label="Train Loss")
ax1.plot(validation_counter, validation_losses, color="tab:orange", alpha=0.7, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Total Energy", color="black")
ax1.tick_params(axis="y")
plt.legend()

fig = plt.figure()
ax2 = fig.gca()
ax2.semilogy(train_counter, train_grads, color="tab:red", alpha=0.7, label="Grad Norm")
ax2.set_ylabel("Gradient Norm", color="tab:red")
ax2.set_xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()