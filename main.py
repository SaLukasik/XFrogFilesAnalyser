import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pandas.errors import ParserWarning


def plot_data(main_folder, sub_folders, **kwargs):
    for i in range(len(sub_folders)):

        df = pd.read_table(fr'{main_folder}\{sub_folders[i]}\Speck.dat', sep="\s+")
        d_array = df.to_numpy()
        plt.figure(1)
        plt.plot(d_array[:, 0], d_array[:, 2], label=sub_folders[i])

        plt.figure(2)
        plt.plot(d_array[:, 0], d_array[:, 1], label=sub_folders[i])

    parent_dir = main_folder.replace(os.sep, '/')
    path = os.path.join(parent_dir, 'results')
    try:
        os.mkdir(path)
    except OSError as error:
        print('Error during saving, may be not important (is thrown if saving again in same folder) but check if plots are saved')
    plt.figure(1)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Phase [rad]')
    plt.xlim(kwargs.get('cut_phas'))
    plt.legend()
    plt.grid()
    plt.savefig(fr"{main_folder}\results\wave_and_phase.pdf")

    plt.figure(2)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.xlim(kwargs.get('cut_int'))
    plt.grid()
    plt.legend()
    plt.savefig(fr"{main_folder}\results\wave_and_intensity.pdf")
    plt.show()

def plotter(df):

    pixels = df.to_numpy()[:,0]
    size = int(df.columns[0][0])
    min_val = float(df.columns[0][1])
    max_val = float(df.columns[1][1])
    wavelenghts = pixels[0:size]
    wavelenghts = wavelenghts.astype('float32')
    delay_values = pixels[size:2*size]
    delay_values = delay_values.astype('float32')
    data = pixels[2*size:]
    data = data.astype('float32')
    data[data == 0] = 0.00000001

    #data = np.log(data)
    #data = np.log((data - min_val)/(max_val-min_val))
    #data = np.log((data - min_val) / (max_val - min_val))
    #print(np.max(data), data.shape)
    #print(len(wavelenghts), len(delay_values), len(data))
    data = data.reshape([size, size])

    #print(type(data), data.shape)
    extender = [np.amin(delay_values), np.amax(delay_values),np.amin(wavelenghts),np.amax(wavelenghts)]
    ratio = abs((np.amin(delay_values) - np.amax(delay_values))/(np.amin(wavelenghts)-np.amax(wavelenghts)))
    ax = plt.imshow(data, interpolation='nearest',  cmap='nipy_spectral', norm=colors.LogNorm(), extent=extender, aspect=ratio, origin = 'lower')
    plt.colorbar(ax)
    plt.xlabel("time delay [fs]")
    plt.ylabel("wavelength [nm]")


def plot_drawings(main_folder, sub_folders, **kwargs):
    for i in tqdm(range(len(sub_folders))):
        warnings.filterwarnings("ignore", category=ParserWarning)
        df = pd.read_table(fr'{main_folder}\{sub_folders[i]}\a.dat', sep="\s+", header = [0,1])
        df2 = pd.read_table(fr'{main_folder}\{sub_folders[i]}\arecon.dat', sep="\s+", header = [0,1])
        fig = plt.figure(figsize=(15, 7))
        fig.add_subplot(1, 2, 1)
        plotter(df)
        plt.title("Orginal")
        fig.add_subplot(1, 2, 2)
        plt.title("Retrieved")
        plotter(df2)
        fig.tight_layout(pad=10.0)

        parent_dir = main_folder.replace(os.sep, '/')
        path = os.path.join(parent_dir, 'results/plots')
        try:
            os.mkdir(path)
        except OSError as error:
            print(
                'Error during saving, may be not important (is thrown if saving again in same folder) but check if plots are saved')

        plt.savefig(fr"{main_folder}\results\plots\{sub_folders[i]}.pdf")
        plt.clf()

if __name__ == '__main__':
    path = r"D:\XFROG\Surowe XFROGi\FROG_SAM_LASER_0fs2_BBO_0.5mm"
    folders = [13, 17, 21, 25, 29]
    plot_data(path, folders, cut_int=(1000, 1050), cut_phas=(900, 1150))
    plot_drawings(path, folders)


