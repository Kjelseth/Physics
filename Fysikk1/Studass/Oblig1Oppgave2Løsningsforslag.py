import numpy as np
import matplotlib.pyplot as plt

# Parametre
M = 100e-3 # Kulens masse  [kg]
C = 5.00e-2 # Luftmotstand konstant  [kg/m]
V0 = 0 # Starthastighet  [m/s]
H0 = 50 # Starthøyde  [m]
G = 9.81 # Tyngdeakselerasjon  [m/s^2]
T_START = 0.0 # Start tid for beregninger  [s]
T_STOP = 15.0 # Slut tid for beregninger  [s]
N = 100 # Antall punkter i simulering/graf

def a(v): # Akselerasjon Positiv retning oppover
    return - G + C/M * v**2
    # Alternativt: - G - C/M * v * np.abs(v)
    # Dette gir alltid luftmotstand motvirkende fartsretning

t = np.linspace(T_START, T_STOP, N)

dt = t[1] - t[0] # Tidssteg størrelse

h = np.zeros(N)
v = np.zeros(N)

h[0] = H0
v[0] = V0

for i in range(N - 1):
    h[i + 1] = h[i] + v[i] * dt
    v[i + 1] = v[i] + a(v[i]) * dt



h_uten = H0 - 0.5 * G * t**2
v_uten = - G * t

for i in range(N):
    if h[i] <= 0:
        t_hit = t[i]
        v_hit = v[i]
        index = i
        break

for i in range(N):
    if h_uten[i] <= 0:
        t_hit_uten = t[i]
        index_uten = i
        break

print(f"Treffer bakken med luftmotstand ved t = {t_hit:.3f}")
print(f"Treffer bakken uten luftmotstand ved t = {t_hit_uten:.3f}")






plt.figure(figsize=(9, 6))
#plt.subplot(2, 1, 1)
plt.grid()

plt.plot(t[:index], h[:index], label="h(t) med luftmotstand")
plt.plot(t[:index_uten], h_uten[:index_uten], label="h(t) uten luftmotstand")

plt.ylabel("Høyde over bakken  [m]")
plt.title("Høyden til kule som slippes")
plt.legend(loc="upper right")

plt.figure(figsize=(9, 6))
#plt.subplot(2, 1, 2)
plt.grid()

plt.plot(t[:index], v[:index], label="v(t) med luftmotstand")
plt.plot(t[:index_uten], v_uten[:index_uten], label="v(t) uten luftmotstand")

plt.xlabel("Tid  [s]")
plt.ylabel("Hastighet  [m/s]")
plt.title("Hastighet til kule som slippes")
plt.legend(loc="lower right")

plt.show()