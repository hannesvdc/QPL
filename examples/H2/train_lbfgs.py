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
N_train = 100_000
B_val = 10
N_validation = 100_000

# Load the best adam model
adam_model_name = 'anti'
model_weights = pt.load( store_directory / f"{adam_model_name}_model_adam.pth", weights_only=True, map_location=device )

# Setup the network
R_cutoff = 5.0
z = 64
neurons_per_layer = [ 1+6*3, z, z, z, z, 1]
model = QuantumNetwork( neurons_per_layer, R_cutoff )
model.to( device=device, dtype=dtype )
model.load_state_dict( model_weights )
print('Number of Trainable Parameters: ', sum( [ p.numel() for p in model.parameters() if p.requires_grad ]))

# Loss function.
chunk_size = 1
loss_fcn = EnergyLoss( chunk_size=chunk_size )

# Setup the optimizer and learning rate scheduler
optimizer = optim.LBFGS( model.parameters(), line_search_fn='strong_wolfe' )

# Main training routine
train_counter : List = []
train_losses : List = []
train_grads : List = []

# Sample a new dataset every time
R, r1, r2, mc_weights = sampleBatchUniformBall( B, N_train, R_cutoff, antithetic=True, gen=gen, device=device, dtype=dtype )
def closure( ) -> pt.Tensor:
    model.train( )
    optimizer.zero_grad( set_to_none=True )
        
    # Compute the loss (backward is called per-chunk inside loss_fcn)
    loss = loss_fcn( model, R, r1, r2, mc_weights, training=True )

    return loss

# Also some validation metrics
B_val = 10
validation_counter : List = []
validation_losses : List = []
val_R = 0.70055 * pt.ones( (B_val,), dtype=dtype, device=device)

# Main training loop
best_val_loss = math.inf
n_epochs = 500
try:
    for epoch in range( 1, n_epochs+1 ):

        train_loss = optimizer.step( closure )
        loss_grad = getGradientNorm( model )

        # Keep track of important metrics
        train_counter.append( epoch-1 )
        train_losses.append( train_loss )
        train_grads.append( float(loss_grad.detach().cpu().item()) )

        # Print some diagnostics
        print_str = (
            f"\nEpoch {epoch:04d} "
            f"Loss: {train_loss:.5e}  "
            f"Grad: {loss_grad.item():.3e}  "
        )
        print(print_str)

        # Independent and random validation samples
        _, val_r1, val_r2, val_mc_weights = sampleBatchUniformBall( B_val, N_validation, R_cutoff, antithetic=True, gen=gen, device=device, dtype=dtype )
        total_energy_mean = loss_fcn( model, val_R, val_r1, val_r2, val_mc_weights, training=False )
        electron_energy = total_energy_mean - 1.0 / (2.0 * float(val_R.item()) )
        print_str = f'\nValidation Epoch {epoch:03d}: \tElectron Energy: {electron_energy:.5e} \tTotal Energy {total_energy_mean:.5e}'
        print( print_str )

        # Store the current model and optimizer weights.
        pt.save( model.state_dict(), store_directory / f"{name}_model_lbfgs.pth")
        pt.save( optimizer.state_dict(), store_directory / f"{name}_optimizer_lbfgs.pth")
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
np.save( store_directory / f"{name}_train_data_lbfgs.npy", train_data)
np.save( store_directory / f"{name}_validation_data_lbfgs.npy", validation_data)

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