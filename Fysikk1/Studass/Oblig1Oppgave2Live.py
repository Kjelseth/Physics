import numpy as np
import matplotlib.pyplot as plt

# Parameter/Konstanter
M = 100e-3 # Masse  [kg]
C = 0.050 # Luftmotstand konstandt  [kg/m]
G = 9.81 # Tyngdeakselerasjon  [m/s^2]
H0 = 50 # Start høyde  [m]
V0 = 0 # Start hastighet [m/s]
A0 = - G # Start aksellerasjon  [m/s^2]
T_START = 0 # Start tid for simulering  [s]
T_STOP = 100 # Slutt tid for simulering  [s]
dt = 1e-3 # Antall iterasjoner

t = np.arange(T_START, T_STOP, dt)

h = np.zeros(len(t))
v = np.zeros(len(t))
a = np.zeros(len(t))

h[0] = H0
v[0] = V0
a[0] = A0

for i in range(len(t) - 1):
    h[i + 1] = h[i] + v[i] * dt
    v[i + 1] = v[i] + a[i] * dt
    a[i + 1] = - G - (C / M) * v[i + 1] * np.abs(v[i + 1])

h_uten = H0 + V0 * t - 0.5 * G * t**2
v_uten = V0 - G * t
a_uten = np.zeros(len(t)) - 1 * G

index_bakke = None
for i in range(len(t)):
    if h[i] <= 0:
        index_bakke = i
        break

index_bakke_uten = None
for i in range(len(t)):
    if h_uten[i] <= 0:
        index_bakke_uten = i
        break

if index_bakke_uten is not None:
    t_uten = t[:index_bakke_uten + 1]
    h_uten = h_uten[:index_bakke_uten + 1]
    v_uten = v_uten[:index_bakke_uten + 1]
    a_uten = a_uten[:index_bakke_uten + 1]

if index_bakke is not None:
    t = t[:index_bakke + 1]
    h = h[:index_bakke + 1]
    v = v[:index_bakke + 1]
    a = a[:index_bakke + 1]
    print(f"Treffer bakken ved t = {t[-1]:.2f} s, ved hastighet {v[-1]:.2f} m/s")
else:
    print(f"Treffer ikke bakken")

#print(f"Hastighet med luftmotstand ved t = {t[100]:.3f} er {v[100]:.3f} m/s")
#print(f"Hastighet uten luftmotstand ved t = {t[100]:.3f} er {v_uten[100]:.3f} m/s")

plt.figure(figsize=(9, 12))

plt.subplot(3, 1, 1)
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(t, h, label="h(t) med luftmotstand")
plt.plot(t_uten, h_uten, label="h(t) uten luftmotstand")

if index_bakke is not None:
    plt.plot(t[-1], h[-1], marker="o", label=f"treffer ved t = {t[-1]:.1f} s, med luftmotstand")

if index_bakke_uten is not None:
    plt.plot(t_uten[-1], h_uten[-1], marker="o", label=f"treffer ved t = {t_uten[-1]:.1f} s, uten luftmotstand")

#plt.xlabel("Tid  [s]")
plt.ylabel("Høyde  [m]")
plt.title("Høyde, hastighet og akselerasjon når ball slippes")
plt.legend()

plt.subplot(3, 1, 2)
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(t, v, label="v(t) med luftmotstand")
plt.plot(t_uten, v_uten, label="v(t) uten luftmotstand")

if index_bakke is not None:
    plt.plot(t[-1], v[-1], marker="o", label=f"treffer med v = {v[-1]:.1f} m/s, med luftmotstand")

if index_bakke_uten is not None:
    plt.plot(t_uten[-1], v_uten[-1], marker="o", label=f"treffer med v = {v_uten[-1]:.1f} m/s, uten luftmotstand")

#plt.xlabel("Tid  [s]")
plt.ylabel("Hastighet  [m/s]")
plt.legend()


plt.subplot(3, 1, 3)
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(t, a, label="a(t) med luftmotstand")
plt.plot(t_uten, a_uten, label="a(t) uten luftmotstand")

if index_bakke is not None:
    plt.plot(t[-1], a[-1], marker="o", label=f"treffer med a = {a[-1]:.1f} m/s^2, med luftmotstand")

if index_bakke_uten is not None:
    plt.plot(t_uten[-1], a_uten[-1], marker="o", label=f"treffer med a = {a_uten[-1]:.1f} m/s^2, uten luftmotstand")

plt.xlabel("Tid  [s]")
plt.ylabel("Akselerasjon  [m/s^2]")
plt.legend()

plt.show()