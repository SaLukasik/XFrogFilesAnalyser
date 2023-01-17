import os
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pandas.errors import ParserWarning
import shutil


def plot_im(df, fiber='none'):
    pixels = df.to_numpy()[:, 0]
    size = int(df.columns[0][0])
    print(size)
    print(pixels)
    min_val = float(df.columns[0][1])
    max_val = float(df.columns[1][1])
    wavelengths = pixels[0:size]
    wavelengths = wavelengths.astype('float32')


    delay_values = pixels[size:2 * size]
    delay_values = delay_values.astype('float32')
    data = pixels[2 * size:]
    data = data.astype('float32')
    data[data == 0] = 0.00000001

    # data = np.log(data)
    # data = np.log((data - min_val)/(max_val-min_val))
    # data = np.log((data - min_val) / (max_val - min_val))
    # print(np.max(data), data.shape)
    # print(len(wavelenghts), len(delay_values), len(data))
    data = data.reshape([size, size])
    start = np.where(wavelengths >= 509)[0]
    end = np.where(wavelengths <= 523)[0]
    wavelengths = wavelengths[start[0]:end[-1]]
    data = data[start[0]:end[-1], :]
    # cmap
    crange = 1000
    my_map = mpl.colormaps['jet'].resampled(crange)
    my_colors = my_map(range(crange))
    white = np.array([[1,1,1,1]])
    new_colormap = ListedColormap(np.concatenate((white, my_colors)))
    #

    # print(type(data), data.shape)
    extender = [np.amin(delay_values), np.amax(delay_values), np.amin(wavelengths), np.amax(wavelengths)]
    ratio = abs((np.amin(delay_values) - np.amax(delay_values)) / (np.amin(wavelengths) - np.amax(wavelengths)))
    ax = plt.imshow(data, interpolation='bicubic', cmap=new_colormap,  extent=extender,
                    aspect=ratio, origin='lower')
    # plt.colorbar(ax)
    plt.xlabel("time delay [fs]")
    plt.ylabel("wavelength [nm]")
    plt.axhline(520, color='r', linestyle='--', linewidth=0.5)



def plot_spec(sp, fiber='none'):
    d_array = sp.to_numpy()
    # finding index max and min
    start = np.where(d_array[:, 0] <= 1070)[0]
    end = np.where(d_array[:, 0] >= 1000)[0]

    d_array = d_array[start[0]:end[-1]]
    # plt.plot(d_array[:, 0], d_array[:, 1])
    plt.plot(d_array[:, 0], np.log(d_array[:, 1]))

    plt.axvline(1040, color='r', linestyle='--', linewidth=0.5)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('normalised intensity')
    plt.grid()
    pass


fig = plt.figure(1)
warnings.filterwarnings("ignore", category=ParserWarning)
# DATA_DOWNLOAD
speck_path = r"D:\XFROG\Surowe XFROGi\SMF28_160mm_A110x2_-48fs2_BBO_0.5mm\27\Speck.dat"
frog_path = r"D:\XFROG\Surowe XFROGi\SMF28_160mm_A110x2_-48fs2_BBO_0.5mm\27\a.dat"
sp = pd.read_table(speck_path, sep="\s+")
df = pd.read_table(frog_path, sep="\s+", header=[0, 1])
#
fig.add_subplot(2, 3, 1)
plt.title('SMF28, 27% pow')
plot_im(df, fiber='smf28')
fig.add_subplot(2, 3, 4)
plot_spec(sp)

speck_path = r"D:\XFROG\Surowe XFROGi\NV4A1_160mm_A110x2_-48fs2_BBO_0.5mm_800nm\31\Speck.dat"
frog_path = r"D:\XFROG\Surowe XFROGi\NV4A1_160mm_A110x2_-48fs2_BBO_0.5mm_800nm\31\a.dat"
sp = pd.read_table(speck_path, sep="\s+")
df = pd.read_table(frog_path, sep="\s+", header=[0, 1])
fig.add_subplot(2, 3, 2)
plt.title('NV4A1, 31% pow')
plot_im(df)
fig.add_subplot(2, 3, 5)
plot_spec(sp)
#
speck_path = r"D:\XFROG\Surowe XFROGi\NDF7T1_159mm_A110x2_-48fs2_BBO_0.5\31\Speck.dat"
frog_path = r"D:\XFROG\Surowe XFROGi\NDF7T1_159mm_A110x2_-48fs2_BBO_0.5\31\a.dat"
sp = pd.read_table(speck_path, sep="\s+")
df = pd.read_table(frog_path, sep="\s+", header=[0, 1])
#
fig.add_subplot(2, 3, 3)
plt.title('NDF7T1, 31% pow')
plot_im(df)
fig.add_subplot(2, 3, 6)
plot_spec(sp)
#
plt.subplots_adjust(left=0.07,
                    bottom=0.1,
                    right=0.98,
                    top=0.95,
                    wspace=0.5,
                    hspace=0.4)
plt.show()
