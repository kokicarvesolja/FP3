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

#konstante 
u_0 = 1.256e-6
S, ΔS = 4e-2**2, 2*4e-2 * 0.5e-2
a, Δa = 10.9e-2, 0.5e-2
b, Δb = 13e-2, 1e-2
L, ΔL = 2*a + 2*b, 2*Δa + 2*Δb
d, Δd = 1.68e-3/18, 0.01e-3/18
N, n = 1000, 46

def llseg_intersect(l, A, B):
    l_ = np.cross(np.append(A, 1), np.append(B, 1))
    x, y, α = np.cross(l, l_)
    I = np.array([x, y]) / α
    AB, AI = B - A, I - A
    if 0 < np.dot(AB, AI) <= np.dot(AB, AB):
        return I
    return None

def lcurve_intersect(l, x_y):
    for i in range(len(x_y) - 1):
        A, B = x_y[i:i+2]
        I = llseg_intersect(l, A, B)
        if I is not None:
            return I
    return None

data = np.loadtxt("./meritve/01Jeklo.txt")

I, F = data[:, 1], data[:, 3]

H = N / L * I
B = 1 / (S * n) * F
B -= 0.5 * (np.min(B) + np.max(B))

plt.plot(H, B)
plt.title("Histerezna zanka, neprekinjen krog")
plt.xlabel("$H[Am^{-1}]$")
plt.ylabel("$B[T]$")
plt.grid()
plt.savefig("jeklo.png")
#plt.show()
plt.close()


k = - u_0 * L / d
H_B = np.stack((H, B), axis=1)
H_B_upper_left = H_B[(B > 0) * (H < 0)]
print(np.nonzero((B > 0) * (H < 0)))
H_intersects, B_intersects = np.array([lcurve_intersect((k, -i, 0), H_B[650:700]) for i in [3, 6, 9, 12, 15, 18]]).T

plt.plot(H, B)
plt.scatter(H_intersects, B_intersects, color='r', marker='x', zorder=3)
H_slit = np.linspace(-180, 180)
plt.plot(3*H_slit, k*H_slit, color='black', linestyle=':', label=r'')
plt.plot(6*H_slit, k*H_slit, color='black', linestyle=':')
plt.plot(9*H_slit, k*H_slit, color='black', linestyle=':')
plt.plot(12*H_slit, k*H_slit, color='black', linestyle=':')
plt.plot(15*H_slit, k*H_slit, color='black', linestyle=':')
plt.plot(18*H_slit[7:-7], k*H_slit[7:-7], color='black', linestyle=':')
plt.xlim(-120, 0)
plt.ylim(0, 0.3)
plt.grid()
plt.xlabel("$H[Am^{-1}]$")
plt.ylabel("$B[T]$")
plt.title("Presecisca premic I=0 z magnetizacijsko krivuljo")
plt.savefig("presecisca.png")
#plt.show()
plt.close()

reze = ["meritve/020Jeklo_3.txt", "meritve/021Jeklo_6.txt", "meritve/022Jeklo_9.txt", "meritve/023Jeklo_12.txt", "meritve/024Jeklo_15.txt", "meritve/025Jeklo_18.txt"]
naslovi = [f'Reža debeline {s}d' for s in [3, 6, 9, 12, 15, 18]]
B_I0 = []
fig, axs = plt.subplots(3, 2, figsize=(6, 9))
for ax, meritve, titles in zip(axs.flatten(), reze, naslovi):
    data = np.loadtxt(meritve)
    I, F = data[:, 1], data[:, 3]
    U_m = N * I
    B = 1 / (S * n) * F
    U_B = np.stack((U_m, B), axis=1)
    U_B_poz = U_B[B > 0]
    tocka_U, tocka_B = lcurve_intersect((1, 0, 0), U_B_poz)
    B_I0.append(tocka_B)
    ax.scatter(tocka_U, tocka_B, color='k', marker='x', zorder=3)
    ax.plot(H, B)
    ax.set_xlabel("$U_m [A]$")
    ax.set_ylabel("$B[T]$")
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.set_title(titles)

fig.tight_layout()
fig.savefig("grafi.png")
plt.close()

data = np.loadtxt("meritve/03Zelezo.txt")

I, F = data[:, 1], data[:, 3]

H = N / L * I
B = 1 / (S * n) * F
B -= 0.5 * (np.min(B) + np.max(B))

plt.plot(H, B)
plt.title("Histerezna zanka, neprekinjen delno zelezen krog")
plt.xlabel("$H[Am^{-1}]$")
plt.ylabel("$B[T]$")
plt.grid()
plt.savefig("zelezo.png")
#plt.show()
plt.close()

print(''.join([f'{i} & {1e3 * B:.1f} & {1e3 * real_B:.1f} & {1e3 * np.abs(real_B - B):.1f} & {np.abs(real_B/B - 1):.1} \\\\\n'
               for i, B, real_B in zip([1, 2, 3, 4, 8, 16], B_intersects, B_I0)]))