import emd
import numpy as np
import matplotlib.pyplot as plt

Array = np.loadtxt("data.txt", dtype=float, delimiter=None, converters=None, skiprows=0, usecols=(0, 1, 2), unpack=False, ndmin=0)

sample_rate = 1

plt.plot(Array[:, 0], Array[:, 1])
plt.title("Original Signal")
plt.xlabel("Days")
plt.ylabel("Counts")
plt.savefig("Original Signal.png")
plt.show()

imf = emd.sift.mask_sift(Array[:, 1], max_imfs=5)

emd.plotting.plot_imfs(imf, xlabel="Days")
plt.savefig("imf", dpi=300)
plt.show()

fft_results = np.absolute(np.fft.rfft(Array[:, 1], n=None, axis=-1, norm=None))**2

freq = np.linspace(0, 0.5, len(fft_results))

peak_index = np.argmax(fft_results[10:]) + 10
peak = freq[peak_index]
print(len(fft_results), peak_index, fft_results[peak_index], freq[peak_index])

plt.axvline(x=peak, color="r")
plt.loglog(freq, fft_results)
plt.text(peak*1.1, 3000, "Peak at\n0.02464/day = 40.58 days")
plt.xlabel("Frequency (1/day)")
plt.ylabel("Power")
plt.title("Data after Fast Fourier Transform")
plt.ylim([6e-5, 6e4])
plt.savefig("FFT.png", dpi=300)
plt.show()

fft_results = np.absolute(np.fft.rfft(imf[:, 3], n=None, axis=-1, norm=None))**2

freq = np.linspace(0, 0.5, len(fft_results))

peak_index = np.argmax(fft_results[10:]) + 10
peak = freq[peak_index]
print(len(fft_results), peak_index, fft_results[peak_index], freq[peak_index])

plt.axvline(x=peak, color="r")
plt.loglog(freq, fft_results)
plt.text(peak*1.1, 3000, "Peak at\n0.02464/day = 40.58 days")
plt.xlabel("Frequency (1/day)")
plt.ylabel("Power")
plt.title("IMF-4 data after Fast Fourier Transform")
plt.ylim([6e-5, 6e4])
plt.savefig("FFT after EMD.png", dpi=300)
plt.show()

IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')

plt.figure(figsize=(8, 4))

plt.hist(IF[:, 0], np.linspace(0, 0.1, 1000), weights=IA[:, 0])
plt.hist(IF[:, 1], np.linspace(0, 0.1, 1000), weights=IA[:, 1])
plt.hist(IF[:, 2], np.linspace(0, 0.1, 1000), weights=IA[:, 2])

y, x, _ = plt.hist(IF[:, 3], np.linspace(0, 0.1, 1000), weights=IA[:, 3])
print(x.max(), y.max())
peak = x[np.argmax(y)]

plt.hist(IF[:, 4], np.linspace(0, 0.1, 1000), weights=IA[:, 4])

plt.xlabel('Frequency (1/day)')
plt.ylabel('Power')

plt.axvline(x=peak, color="b", zorder=-10)
plt.text(peak*1.1, 3, "Peak at\n{:.5f}/day = {:.2f} days".format(peak, 1/peak))
plt.title("Hilbert Spectrum")
plt.legend(['Peak', 'IMF-1', 'IMF-2', 'IMF-3', 'IMF-4', 'IMF-5'], frameon=False)

plt.savefig("HHT.png", dpi=300)
plt.show()