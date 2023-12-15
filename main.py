# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_input(path):
    # Use a breakpoint in the code line below to debug your script.
    wave_lengths = []
    intensities = []

    with open(path) as f:
        inputs = f.readlines()

    for input_index in inputs:
        wave_length, axis, intensity = input_index.split(",")
        wave_lengths.append(float(wave_length))
        intensities.append(float(intensity.strip()))

    # print(contents)  # Press Ctrl+F8 to toggle the breakpoint.
    return [wave_lengths, intensities]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wave, intensity = load_input("input.txt")
    plt.xticks([510, 560, 620, 680, 740, 780])
    plt.yticks([0, 1500, 3000, 4500, 6000, 7500])
    plt.plot(np.array(wave), np.array(intensity))
    # plt.subplot(2, 1, 2)
    # plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    #            ["500", "560", "620", "680", "740", "800"])
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8,1.0],
    #            ["0", "1500", "3000", "4500", "6000", "7500"])
    # plt.plot(np.array(normalized_wave), np.array(normalized_intensity))
    n_components = 2
    model = NMF(n_components=n_components, init='random', random_state=0)
    intensity_array = np.array(intensity)
    intensity_array = np.maximum(intensity_array, 0)
    W = model.fit_transform(intensity_array.reshape(-1, 1))
    H = model.components_

    for i in range(n_components):
        plt.plot(wave, W[:, i]*H[i, :])

    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity [arb.u.]')
    plt.show()
