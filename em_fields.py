import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pandas.errors import ParserWarning
import shutil

path = r"D:\XFROG\Surowe XFROGi\FROG_SAM_LASER_-48fs2_BBO_0.5mm"
folder = 13

main_folder = path
sub_folder = folder
df_time = pd.read_table(fr'{main_folder}\{sub_folder}\Ek.dat', sep="\s+", skiprows=1)
d_array_time = df_time.to_numpy()
time = d_array_time[:, 0]
em_real = d_array_time[:, 3]
em_imag = d_array_time[:, 4]


fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(em_real, em_imag, time, c=time, alpha=1)
plt.xlabel("\nReal")
plt.ylabel("\nImaginary")
ax.set_zlabel('\nTime [fs]')


path = r"D:\XFROG\Surowe XFROGi\NDF7T1_159mm_A110x2_-48fs2_BBO_0.5"
folder = 29
main_folder = path
sub_folder = folder
df_time = pd.read_table(fr'{main_folder}\{sub_folder}\Ek.dat', sep="\s+", skiprows=1)
d_array_time = df_time.to_numpy()
time = d_array_time[:, 0]
em_real = d_array_time[:, 3]
em_imag = d_array_time[:, 4]

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(em_real, em_imag, time, c=time, alpha=1)
plt.xlabel("\nReal")
plt.ylabel("\nImaginary")
ax.set_zlabel('\nTime [fs]')

path = r"D:\XFROG\Surowe XFROGi\FROG_SAM_LASER_-48fs2_BBO_0.5mm"
folder = 29
main_folder = path
sub_folder = folder
df_time = pd.read_table(fr'{main_folder}\{sub_folder}\Ek.dat', sep="\s+", skiprows=1)
d_array_time = df_time.to_numpy()
time = d_array_time[:, 0]
em_real = d_array_time[:, 3]
em_imag = d_array_time[:, 4]
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(em_real, em_imag, time, c=time, alpha=1,cmap="magma")
plt.xlabel("\nReal")
plt.ylabel("\nImaginary")
ax.set_zlabel('\nTime [fs]')

path = r"D:\XFROG\Surowe XFROGi\FROG_SAM_LASER_-48fs2_BBO_0.5mm"
folder = 29
main_folder = path
sub_folder = folder
df_time = pd.read_table(fr'{main_folder}\{sub_folder}\Ek.dat', sep="\s+", skiprows=1)
d_array_time = df_time.to_numpy()
time = d_array_time[:, 0]
em_real = d_array_time[:, 3]
em_imag = d_array_time[:, 4]
fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(em_real, em_imag, time, c=time, alpha=1,cmap="magma")
plt.xlabel("\nReal")
plt.ylabel("\nImaginary")
ax.set_zlabel('\nTime [fs]')
plt.show()
