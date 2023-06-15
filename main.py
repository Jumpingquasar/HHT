import emd
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.patches as patches

Array = np.loadtxt("data.txt", dtype=float, delimiter=None, converters=None, skiprows=0, usecols=(0, 1, 2), unpack=False, ndmin=0)

sample_rate = 6250

# plt.plot(Array[:, 0], Array[:, 1])

imf = emd.sift.mask_sift(Array[:, 1], max_imfs=9)

emd.plotting.plot_imfs(imf[:sample_rate, :])

emd.plotting.plot_imfs(imf, xlabel="Days")

IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')

freq_edges, freq_centres = emd.spectra.define_hist_bins(0, 100, 128, 'linear')

# Amplitude weighted HHT per IMF
f, spec_weighted = emd.spectra.hilberthuang(IF, IA, freq_edges, sum_imfs=False)

# Unweighted HHT per IMF - we replace the instantaneous amplitude values with ones
f, spec_unweighted = emd.spectra.hilberthuang(IF, np.ones_like(IA), freq_edges, sum_imfs=False)

plt.figure(figsize=(10, 4))
plt.subplots_adjust(hspace=0.4)
plt.subplot(121)
plt.plot(freq_centres, spec_unweighted)
plt.xticks(np.arange(10)*10)
plt.xlim(0, 100)
plt.xlabel('Frequency (Days)')
plt.ylabel('Count')
plt.title('unweighted\nHilbert-Huang Transform')

plt.subplot(122)
plt.plot(freq_centres, spec_weighted)
plt.xticks(np.arange(10)*10)
plt.xlim(0, 100)
plt.xlabel('Frequency (Days)')
plt.ylabel('Power')
plt.title('IA-weighted\nHilbert-Huang Transform')
plt.legend(['IMF-1', 'IMF-2', 'IMF-3', 'IMF-4', 'IMF-5', 'IMF-6', 'IMF-7', 'IMF-8', 'IMF-9'], frameon=False)

plt.show()