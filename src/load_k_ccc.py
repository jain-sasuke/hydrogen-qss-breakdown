import numpy as np
import pandas as pd

K_exc   = np.load('data/processed/collisions/ccc/K_CCC_exc_table.npy')   # (870, 12)
K_deexc = np.load('data/processed/collisions/ccc/K_CCC_deexc_table.npy') # (870, 12)
Te_grid = np.load('data/processed/collisions/ccc/Te_grid.npy')            # (12,)
meta    = pd.read_csv('data/processed/collisions/ccc/K_CCC_metadata.csv')

# Row idx in tables matches idx column in meta
# meta gives you n_i, l_i, n_f, l_f, omega_i, omega_f, dE_eV for each row