import numpy as np
from scipy.optimize import curve_fit

U = np.array([0.00000675, 0.00001045, 0.00001395, 0.00001780, 0.00002155,
             0.00002530, 0.00002935, 0.00003275, 0.00003605, 0.00003955,
             0.00004360, 0.00004720, 0.00005005, 0.00005345, 0.00005915])  # 15 wartości napięcia w V

I = np.array([ 0.53,  0.76,  0.98,  1.22,  1.46,
               1.69,  1.94,  2.16,  2.35,  2.57,
               2.82,  3.04,  3.21,  3.42,  3.77] )  # 15 wartości natężenia w A

d = np.array([0.02508, 0.02510, 0.02410, 0.02410, 0.02512 ] )  # 5 wartości długości pręta w m
l = np.array([0.41900, 0.42000, 0.42100, 0.42000, 0.42000] )  # 5 wartości średnicy pręta w m

u_B_d = 0.00030  # Niepewność systematyczna średnicy (m)
u_B_l = 0.00040  # Niepewność systematyczna długości (m)
u_B_U = 0.000001  # Niepewność napięcia (V)
u_B_I = 0.16  # Niepewność natężenia (A)

# Obliczenie średnich wartości l i d
l_mean = np.mean(l)
d_mean = np.mean(d)

# Obliczenie niepewności typu A dla d i l
N_d = len(d)
N_l = len(l)
u_A_d = np.sqrt(np.sum((d - d_mean)**2) / (N_d * (N_d - 1)))
u_A_l = np.sqrt(np.sum((l - l_mean)**2) / (N_l * (N_l - 1)))

# Łączenie niepewności A i B
u_d = np.sqrt(u_A_d**2 + u_B_d**2)
u_l = np.sqrt(u_A_l**2 + u_B_l**2)

# Definicja funkcji liniowej przechodzącej przez (0,0)
def linear_model(U, a):
    return a * U

# Dopasowanie modelu do danych
params, _ = curve_fit(linear_model, U, I)
a = params[0]  # Współczynnik nachylenia

# Obliczenie niepewności dla a na podstawie względnych błędów pomiarowych
rel_u_I = u_B_I / np.mean(I)
rel_u_U = u_B_U / np.mean(U)
u_a = a * np.sqrt(rel_u_I**2 + rel_u_U**2)

# Obliczenie pola przekroju poprzecznego A
A = (np.pi * d_mean**2) / 4

# Obliczenie konduktywności
sigma = (4 * l_mean * a) / (np.pi * d_mean**2)

# Współczynniki propagacji niepewności
c_l = sigma / l_mean
c_d = sigma / d_mean
c_a = sigma / a

# Obliczenie całkowitej niepewności konduktywności
u_sigma = np.sqrt((c_l**2) * u_A_l**2 + (c_d**2) * u_B_d**2 + (c_a**2) * u_a**2)

# Wyświetlenie wyników
print(f"Współczynnik a: {a:.6f} ± {u_a:.6f}")
print(f"Średnia średnica d: {d_mean:.6f} m ± {u_d:.6f}")
print(f"Średnia długość l: {l_mean:.6f} m ± {u_l:.6f}")
print(f"Konduktywność: {sigma:.6f} S/m ± {u_sigma:.6f} S/m, {u_sigma/sigma*100:.2f}%")
