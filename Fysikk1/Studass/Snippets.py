import numpy as np
import matplotlib.pyplot as plt

# Plot setup single
plt.figure(figsize=(9, 9))
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(X_axis, Y_axis, label="f(x)", color="#2E0854")

plt.margins(x=0)
plt.xlabel("Lable name")

plt.legend(loc="upper right")
plt.show()

# Plot setupt double
plt.figure(figsize=(9, 9))
plt.subplot(2, 1, 1)
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(X_axis, Y_axis, label="f(x)", color="#2E0854")

plt.legend(loc="upper right")
plt.margins(x=0)

plt.subplot(2, 1, 2)
plt.minorticks_on()
plt.grid(which="major", linewidth=0.75, alpha=0.5)
plt.grid(which="minor", linewidth=0.25, alpha=0.4)

plt.plot(X_axis, Y_axis, label="f(x)", color="#2E0854")

plt.margins(x=0)
plt.xlabel("Lable name")

plt.legend(loc="upper right")
plt.show()