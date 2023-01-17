# shapes to try: gaussian, super gaussian, sech
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd


def gauss(x, sigma, a):
    return a * np.exp(-2 * x ** 2 / (sigma ** 2))


def sech(x, b):
    return 1/(np.cosh(x/b)**2)

path = r"D:\XFROG\Surowe XFROGi\FROG_SAM_LASER_-48fs2_BBO_0.5mm"
folder = 21
df = pd.read_table(fr'{path}\{folder}\Ek.dat', sep="\s+", skiprows=1)
d_array = df.to_numpy()
plt.plot(d_array[:, 0], d_array[:, 1], '.', label='data', color = 'lightskyblue')
plt.ylabel('Intensity')
plt.xlabel('Time [fs]')
plt.grid()
plt.legend()

xdata = np.asarray(d_array[:, 0])
ydata = np.asarray(d_array[:, 1])

parameters, covariance = curve_fit(gauss, xdata, ydata, p0 = (100,1))
fit_y = gauss(xdata,parameters[0], parameters[1])
plt.plot(xdata, fit_y, label = fr'Gaussian fit: I$_0$ = {round(parameters[1]), 3}, $\omega$ = {round(parameters[0],3)} ')

parameters, covariance = curve_fit(sech, xdata, ydata, p0 = (1000))
fit_y = sech(xdata,parameters[0])
plt.plot(xdata, fit_y, label = fr'sech$^2$ fit: P$_0$ = 1, $\tau$ = {round(parameters[0],3)} ')


plt.legend()
first, second, firsti, secondi = 0, 0, 0, 0

for t in range(len(xdata)):
    x = round(sech(xdata[t], parameters[0]),1)
    if xdata[t] < 0 and x == 0.5:
        first = xdata[t]
        firsti = t
    elif x == 0.5:
        second = xdata[t]
        secondi = t

time_width = (abs(first) + second)
time_widthi = abs(abs(firsti) - abs(secondi))
print(first, second, time_width)
print(time_widthi)
plt.show()


