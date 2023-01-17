import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pandas.errors import ParserWarning
import shutil


def prepare_for_x_frog(main_folder, sub_folders, **kwargs):
    for i in tqdm(range(len(sub_folders))):
        file = open(fr'{main_folder}\{sub_folders[i]}\frog.dat', "r+")
        text = file.read()
        idx = text.find("Central wavelength of retrieved field is ")
        idx_wave = len("Central wavelength of retrieved field is ") + idx
        j = 1
        while (text[idx_wave + j] != " "):
            j += 1
        j -= 1
        wavelen = text[idx_wave:idx_wave + 4]
        file.close()

        file2 = open(fr'{main_folder}\{sub_folders[i]}\Ek.dat', "r")
        if len(file2.readline()) > 5:
            copy = file2.read()
            file2.close()

            file3 = open(fr'{main_folder}\{sub_folders[i]}\Ek.dat', "w")
            file3.write(wavelen + "\n")
            file3.write(copy)
            file.close()


def plot_data(main_folder, sub_folders, **kwargs):
    f = kwargs.get('filter')
    start = -1
    stop = -1
    start_t = -1
    stop_t = -1

    for i in range(len(sub_folders)):
        new_start = -1
        new_stop = -1
        new_start_t = -1
        new_stop_t = -1
        df = pd.read_table(fr'{main_folder}\{sub_folders[i]}\Speck.dat', sep="\s+")
        df_time = pd.read_table(fr'{main_folder}\{sub_folders[i]}\Ek.dat', sep="\s+", skiprows=1)
        d_array = df.to_numpy()
        d_array_time = df_time.to_numpy()

        if f is not None:
            for k in range(len(d_array)):
                if d_array[k, 1] <= f:
                    d_array[k, 2] = 0
                elif new_start == -1:
                    new_start = d_array[k, 0]
                if new_stop == -1 and (d_array[-k, 1] > f) and k!= 0:
                    new_stop = d_array[-k, 0]
            for k in range(len(d_array_time)):
                if d_array_time[k, 1] <= f:
                    d_array_time[k, 2] = 0
                elif new_start_t == -1:
                    new_start_t = d_array_time[k, 0]
                if new_stop_t == -1 and (d_array_time[-k, 1] > f) and k!= 0:
                    new_stop_t = d_array_time[-k, 0]
        print('current stops')
        print(new_start, new_stop, new_start_t, new_stop_t)
        t = new_start
        new_start = new_stop
        new_stop = t

        if new_stop_t < stop_t or stop_t == -1:
            stop_t = new_stop_t
        if new_start_t > start_t or start_t == -1:
            start_t = new_start_t
        if start == -1 or new_start > start:
            start = new_start
        if stop == -1 or new_stop < stop:
            stop = new_stop

        plt.figure(1)
        plt.plot(d_array[:, 0], d_array[:, 2], label=sub_folders[i])

        plt.figure(2)
        plt.plot(d_array[:, 0], d_array[:, 1], label=sub_folders[i])
        plt.figure(3)
        plt.plot(d_array_time[:, 0], d_array_time[:, 2], label=sub_folders[i])

        plt.figure(4)
        plt.plot(d_array_time[:, 0], d_array_time[:, 1], label=sub_folders[i])
    print( start, stop,start_t,stop_t )
    parent_dir = main_folder.replace(os.sep, '/')
    path = os.path.join(parent_dir, 'results')
    try:
        os.mkdir(path)
    except OSError as error:
        print(
            'Error during saving, may be not important (is thrown if saving again in same folder) but check if plots are saved')
    plt.figure(1)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Phase [rad]')
    if f is not None and kwargs.get('cut_phas') is None:
        plt.xlim((start, stop))
    else:
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

    plt.figure(3)
    plt.xlabel('Time [fs]')
    plt.ylabel('Phase [rad]')
    if f is not None and kwargs.get('cut_phas_t') is None:
        print(start_t, stop_t)
        plt.xlim((start_t, stop_t))
    else:
        plt.xlim(kwargs.get('cut_phas_t'))
    plt.legend()
    plt.grid()
    plt.savefig(fr"{main_folder}\results\time_and_phase.pdf")

    plt.figure(4)
    plt.xlabel('Time [fs]')
    plt.ylabel('Intensity')
    plt.xlim(kwargs.get('cut_int_t'))
    plt.grid()
    plt.legend()
    plt.savefig(fr"{main_folder}\results\time_and_intensity.pdf")

    plt.show()


def plotter(df):
    pixels = df.to_numpy()[:, 0]
    size = int(df.columns[0][0])
    print(size)
    print(pixels)
    min_val = float(df.columns[0][1])
    max_val = float(df.columns[1][1])
    wavelenghts = pixels[0:size]
    wavelenghts = wavelenghts.astype('float32')
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

    # print(type(data), data.shape)
    extender = [np.amin(delay_values), np.amax(delay_values), np.amin(wavelenghts), np.amax(wavelenghts)]
    ratio = abs((np.amin(delay_values) - np.amax(delay_values)) / (np.amin(wavelenghts) - np.amax(wavelenghts)))
    ax = plt.imshow(data, interpolation='nearest', cmap='nipy_spectral', norm=colors.LogNorm(), extent=extender,
                    aspect=ratio, origin='lower')
    plt.colorbar(ax)
    plt.xlabel("time delay [fs]")
    plt.ylabel("wavelength [nm]")


def plot_drawings(main_folder, sub_folders, **kwargs):
    for i in tqdm(range(len(sub_folders))):
        warnings.filterwarnings("ignore", category=ParserWarning)
        df = pd.read_table(fr'{main_folder}\{sub_folders[i]}\a.dat', sep="\s+", header=[0, 1])
        df2 = pd.read_table(fr'{main_folder}\{sub_folders[i]}\arecon.dat', sep="\s+", header=[0, 1])
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


def prepare_folders(main_folder, sub_folders, **kwargs):
    parent_dir = main_folder.replace(os.sep, '/')
    for i in tqdm(range(len(sub_folders))):
        path = os.path.join(parent_dir, f'{sub_folders[i]}')
        try:
            os.mkdir(path)
        except OSError as error:
            print(
                'Error during saving, may be not important (is thrown if saving again in same folder) but check if folders were made')

        # try to make files from **kwarg
        for j in kwargs.get('names'):
            print(fr"\{sub_folders[i]}{j}")
            shutil.copy(fr'{main_folder}\{sub_folders[i]}{j}', fr'{main_folder}\{sub_folders[i]}\{sub_folders[i]}{j}')
            # os.system(fr"copy {main_folder}\11.png {main_folder}\11")


if __name__ == '__main__':
    # path - path to the folder with subfolders containing data from frog or xfrog
    # each subfolder should be named as int number and contain the .dat files from doing
    # frog or x-frog procedure. In this folder program will create "result" folder
    # with the results of the analysis. All previous results will be overwritten!!!
    path = r"D:\XFROG\Surowe XFROGi\NDF7T1_159mm_A110x2_-48fs2_BBO_0.5\x_frog_with_fitted_data_and_phasecut"
    # list of folders to be analysed (each number is folder name)
    folders = [11, 23]

    # "prepare_for_x_frog(...)" is a command that should be used to modify Ek.dat
    # files in given folders, so they could be used for x-frog procedure.
    # prepare_for_x_frog(path, folders)

    # this function will draw 4 plots (phase/intensity on wavelength/time)
    # all plots will be saved/overwritten in "result" folder.
    plot_data(path, folders, cut_int=(1005, 1065), cut_int_t=(-2000, 2000), filter=0.1)

    # this function will draw the scan picture of impulse for impulses
    # from given folders and save them in "results/plots". A plot will be done
    # for each folder separately.

    plot_drawings(path, folders)

    # path_funtion creates folder structure needed for frog files
    # it creates folders with names from given list and then to each folder
    # it copies the files with same number. We need do input the extensions
    # of files we want to copy in "names" agrument.
    # path_frog = r"D:\XFROG\Surowe XFROGi\SMF28_160mm_A110x2_-48fs2_BBO_0.5mm"
    # folders_frog = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    # prepare_folders(path_frog, folders_frog, names=[".txt"])
