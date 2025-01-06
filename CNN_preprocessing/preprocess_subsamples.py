import pathlib as pl
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

# save_dir = pl.Path('./saves')
# out_dir = pl.Path('./saves')
tile_example_path = r'C:\Users\hausw001\surfdrive - Hauswirth, S.M. (Sandra)@surfdrive.surf.nl\Data\GLOBGM\input\tiles_input\tile_048-163\transient'
save_dir = pl.Path('%s/cnn_samples_600' % tile_example_path)
out_dir = pl.Path('%s/cnn_samples_600' % tile_example_path)
training_fraction = 0.1
validation_fraction = 0.1
testing_fraction = 0.1

samples_file = save_dir / 'samples.csv'
samples = pd.read_csv(samples_file, index_col=0)
nsamples = samples.shape[0]

training_indices = np.random.choice(samples.index,
                                    int(nsamples * training_fraction),
                                    replace=False)
training_samples = samples.loc[training_indices]
training_samples.to_csv(out_dir / 'training_samples.csv')
samples = samples.drop(training_indices)

validation_indices = np.random.choice(samples.index,
                                      int(nsamples * validation_fraction),
                                      replace=False)
validation_samples = samples.loc[validation_indices]
validation_samples.to_csv(out_dir / 'validation_samples.csv')
samples = samples.drop(validation_indices)

testing_indices = np.random.choice(samples.index,
                                   int(nsamples * testing_fraction),
                                   replace=False)
testing_samples = samples.loc[testing_indices]
testing_samples.to_csv(out_dir / 'testing_samples.csv')
samples = samples.drop(testing_indices)
