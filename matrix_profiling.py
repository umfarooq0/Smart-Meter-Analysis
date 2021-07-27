# In order to conduct matrix profiling approach, we first need to know what timeframe each smart meter
# actually incapsulates. 

# what we aim to do is develop features using matrix profile, and then use algorithms(FLOSS) to segment



import pandas as pd
import matrixprofile as mp
import matplotlib.pyplot as plt
import stumpy
import numpy as np
from matplotlib.patches import Rectangle

# test id = 'MAC000002'. Has 504 days entry between min and max

df = pd.read_csv('/home/usman/Documents/Smart-Meter-analysis/archive/halfhourly_dataset/halfhourly_dataset/block_0.csv')
df.columns = ['LCLid','tstp','energy']
test_id = df[df.LCLid == 'MAC000002' ]

test_id = test_id[test_id.energy != 'Null']
test_id['energy'] = test_id.energy.astype(float)

windows = [
  
    ('4 Hours', 8),
    ('8 Hours', 16),
    ('24 Hours', 48),
    ('Weekly', 336)
]

profiles = {}

for label, window_size in windows:
    profile = mp.compute(test_id['energy'].values, window_size)
    key = '{} Profile'.format(label)
    profiles[key] = profile

fig, axes = plt.subplots(4,1,sharex=True,figsize=(15,10))

for ax_idx, window in enumerate(windows):
    key = '{} Profile'.format(window[0])
    profile = profiles[key]
    axes[ax_idx].plot(profile['mp'])
    axes[ax_idx].set_title(key)

plt.xlabel('Time')
plt.tight_layout()
plt.show()

window_size = 336
profile = mp.compute(test_id['energy'].values, window_size)
#profile = mp.discover.motifs(profile)
discords_ = mp.discover.discords(profile)
figures = mp.visualize(discords_)
plt.show()
## what do these values mean? when we get the output value, what can we infer from this?
'''

While we are looking at single time series, nad trying to understand/segment features, what
about multiple time series? 

can matrix profiling work when there is a number of timeseries? 

What if we run matrix profiling on each seperate time series, and then this will get us a list of 
features? 


'''



#In order to do segmentation, we need to use stumpy

#window_size = 50  # Approximately, how many data points might be found in a pattern
m = 336

profile = stumpy.stump(test_id['energy'], m=m)

motif_idx = np.argsort(profile[:, 0])[0]

nearest_neighbor_idx = profile[motif_idx, 1]

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

axs[0].plot(test_id['energy'].values)
axs[0].set_ylabel('KwH', fontsize='20')
rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=motif_idx, linestyle="dashed")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
axs[1].plot(profile[:, 0])
plt.show()

windows = [
  
    ('4 Hours', 8),
    ('8 Hours', 16),
    ('24 Hours', 48)
]

profiles = {}



for label, window_size in windows:
    profile = stumpy.stump(test_id['energy'], m=window_size)
    key = '{} Profile'.format(label)
    profiles[key] = profile



subseq_len = 8
cac, regime_locations = stumpy.fluss(profile[:, 1],
                                                   L=subseq_len,
                                                   n_regimes=4,
                                                   excl_factor=1
                                                  )

fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
axs[0].plot(range(test_id['energy'].shape[0]), test_id['energy'])
axs[0].axvline(x=regime_locations[0], linestyle="dashed")
axs[1].plot(range(cac.shape[0]), cac, color='C1')
axs[1].axvline(x=regime_locations[0], linestyle="dashed")
plt.show()