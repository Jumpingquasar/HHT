import emd
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.patches as patches

Array = np.loadtxt("data.txt", dtype=float, delimiter=None, converters=None, skiprows=0, usecols=(0, 1, 2), unpack=False, ndmin=0)

sample_rate = 1

# plt.plot(Array[:, 0], Array[:, 1])
# plt.show()

imf = emd.sift.mask_sift(Array[:, 1], max_imfs=9)

fft_results = np.fft.rfft(Array[:, 1], n=None, axis=-1, norm=None)

# plt.xlim(0, 100)
plt.plot(fft_results)

plt.show()

# emd.plotting.plot_imfs(imf, xlabel="Days")
#
# IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
#
# plt.figure(figsize=(8, 4))
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 0], np.linspace(0, 0.1, 1000), weights=IA[:, 0])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 1], np.linspace(0, 0.1, 1000), weights=IA[:, 1])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 2], np.linspace(0, 0.1, 1000), weights=IA[:, 2])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 3], np.linspace(0, 0.1, 1000), weights=IA[:, 3])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 4], np.linspace(0, 0.1, 1000), weights=IA[:, 4])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 5], np.linspace(0, 0.1, 1000), weights=IA[:, 5])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 6], np.linspace(0, 0.1, 1000), weights=IA[:, 6])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 7], np.linspace(0, 0.1, 1000), weights=IA[:, 7])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# # Plot an amplitude-weighted histogram using frequency bins from 0-20Hz
# plt.hist(IF[:, 8], np.linspace(0, 0.1, 1000), weights=IA[:, 8])
# plt.grid(True)
# plt.title('IF Histogram\nweighted by IA')
# plt.xlabel('Frequency (Hz)')
#
# freq_edges, freq_centres = emd.spectra.define_hist_bins(0, 0.1, 512, 'linear')
#
# # Amplitude weighted HHT per IMF
# f, spec_weighted = emd.spectra.hilberthuang(IF, IA, freq_edges, sum_imfs=False)
#
# # Unweighted HHT per IMF - we replace the instantaneous amplitude values with ones
# f, spec_unweighted = emd.spectra.hilberthuang(IF, np.ones_like(IA), freq_edges, sum_imfs=False)
#
# index = np.argmax(spec_weighted[:, 3])
# found_freq = freq_centres[index]
# print(index, spec_weighted[index, 3], found_freq)
# plt.figure(figsize=(10, 4))
# plt.subplots_adjust(hspace=0.4)
# plt.subplot(121)
# plt.plot(freq_centres, spec_unweighted)
# plt.xticks(np.arange(0, 0.1, 0.01))
# plt.xlim(0, 0.1)
# plt.xlabel('Frequency (1/day)')
# plt.ylabel('Count')
# plt.title('unweighted\nHilbert-Huang Transform')
#
# plt.subplot(122)
# plt.plot(freq_centres, spec_weighted)
# plt.xticks(np.arange(0, 0.1, 0.01))
# plt.xlim(0, 0.100)
# plt.xlabel('Frequency (1/day)')
# plt.ylabel('Power')
# plt.title('IA-weighted\nHilbert-Huang Transform')
# plt.legend(['IMF-1', 'IMF-2', 'IMF-3', 'IMF-4', 'IMF-5', 'IMF-6', 'IMF-7', 'IMF-8', 'IMF-9'], frameon=False)
# plt.savefig("6250.png", dpi=300)
# plt.show()
