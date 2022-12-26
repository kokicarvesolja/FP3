import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
import array_to_latex as a2l
import csv
import pandas as pd

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

data = np.loadtxt('./Meritve/01OdzivNeduseno_KCP.txt')

plt.rcParams['lines.markersize'] = 1

plt.plot(data[:, 0], data[:, 1], 'ro', label=r'A')
plt.plot(data[:, 0], data[:, 2], 'bo', label=r'$\sigma_{A}$') #da mi upošteva supscripte, uporabim argument r = raw
plt.plot(data[:, 0], data[:, 3], 'ko', label=r'$A_{\nu}$')
plt.xlabel(r'Frekvenca $\nu$ [Hz]')
plt.ylabel('Amplituda A')
plt.legend()
plt.title('Resonancni odziv akusticnega resonatorja brez absorberja')
#plt.savefig('01Neduseno')
plt.show()
plt.close()

data = np.loadtxt('Meritve/03OdzivDuseno2HzStep_KCP.txt')

plt.plot(data[:, 0], data[:, 1], 'ro', label=r'A')
plt.plot(data[:, 0], data[:, 2], 'bo', label=r'$\sigma_{A}$')
plt.plot(data[:, 0], data[:, 3], 'ko', label=r'$A_{\nu}$')
plt.xlabel(r'Frekvenca $\nu$ [Hz]')
plt.ylabel('Amplituda A')
plt.legend()
plt.title('Resonancni odziv akusticnega resonatorja z absorberjem')
#plt.savefig('02Duseno')
#plt.show()
plt.close()

def zvocni_odziv(datoteka, frekv, ime):
    data = np.loadtxt(datoteka)

    os_x = np.linspace(0, 56.7, len(data))

    plt.plot(os_x, data[:, 0], 'ro', label=r'A')
    plt.plot(os_x, data[:, 1], 'bo', label=r'$\sigma_{A}$')
    plt.plot(os_x, data[:, 2], 'ko', label=r'$A_{\nu}$')
    plt.xlabel('Razdalja r [cm]')
    plt.ylabel('Amplituda A')
    plt.legend()
    plt.title('Zvocni profil v resonatorju ' + frekv)
    #plt.savefig(ime)
    #plt.show()
    plt.close()
    
    pass

zvocni_odziv('Meritve/041ZvocniProfil308Hz_KCP.txt', '308Hz', '03308Hz')
zvocni_odziv('Meritve/042ZvocniOdziv607Hz_KCP.txt', '607Hz', '04607Hz')
zvocni_odziv('Meritve/043ZvocniOdziv911Hz_KCP.txt', '911Hz', '05911Hz')
zvocni_odziv('Meritve/044ZvocniOdziv450Hz_KCP.txt', '450Hz', '06450Hz')

A = unp.uarray([0.567], [0.001])

def i_am_speed(frek, x, B):
    return 2 * frek / unp.sqrt((x / B) ** 2)

print(i_am_speed(911, 3, A))

#hitrosti = unp.uarray([i_am_speed(308, 1, A), i_am_speed(607, 2, A), i_am_speed(911, 3, A)])

#print(np.average(hitrosti), hitrosti - np.average(hitrosti))

#print(print(a2l.to_ltx(hitrosti, frmt='{:6.3f}', arraytype='array')))