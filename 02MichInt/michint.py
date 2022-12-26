import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
import array_to_latex as a2l
import csv
import pandas as pd
from astropy.io.votable import parse
from astropy.table import QTable, Table, Column

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}) #font na grafih je LaTexov
rc('text', usetex=True)

delta = unp.uarray([632.8], [0.1])
x_povp = unp.uarray([0.16], [0.01])

R = 100 * delta * 10 ** (-9) / (2 * x_povp * 10 ** (-3))

l = unp.uarray([50], [1])

data =np.array(pd.read_csv(r'data.csv', delimiter=";"))

delta_n = delta * 10 ** (-9) * data[:, 1] / (2 * l * 10 ** (-3))

def lin_f(x, k, n):
    return k * x + n

koef, _ = curve_fit(lin_f, data[:, 0], unp.nominal_values(delta_n))

tlak = unp.uarray([data[:, 0]], 5 * [0.1])

plt.errorbar(unp.nominal_values(tlak[0]), unp.nominal_values(delta_n), xerr=unp.std_devs(tlak[0]), yerr=unp.std_devs(delta_n), fmt='o', label='Meritve')
plt.plot(unp.nominal_values(tlak[0]), lin_f(unp.nominal_values(tlak[0]), *koef), label="Regresivna premica")
plt.title("Lomni kolicnik v odvisnost od tlaka")
plt.xlabel("Tlak p [bar]")
plt.ylabel("Lomni kolicnik $n_0 - 1$")
plt.legend()
plt.savefig("02.png")
plt.show()
plt.close()



