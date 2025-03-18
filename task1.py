import numpy as np
import matplotlib.pyplot as plt


def lotka_volterra_rk4(a11, a12, a21, a22, x0, y0, h, T):
    num_steps = int(T / h) + 1
    t_values = np.linspace(0, T, num_steps)
    x_values = np.zeros(num_steps)
    y_values = np.zeros(num_steps)

    x_values[0] = x0
    y_values[0] = y0

    def dx_dt(x, y):
        return a11 * x - a12 * x * y

    def dy_dt(x, y):
        return -a22 * y + a21 * x * y

    for i in range(1, num_steps):
        x, y = x_values[i - 1], y_values[i - 1]

        k1_x = h * dx_dt(x, y)
        k1_y = h * dy_dt(x, y)

        k2_x = h * dx_dt(x + k1_x / 2, y + k1_y / 2)
        k2_y = h * dy_dt(x + k1_x / 2, y + k1_y / 2)

        k3_x = h * dx_dt(x + k2_x / 2, y + k2_y / 2)
        k3_y = h * dy_dt(x + k2_x / 2, y + k2_y / 2)

        k4_x = h * dx_dt(x + k3_x, y + k3_y)
        k4_y = h * dy_dt(x + k3_x, y + k3_y)

        x_values[i] = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y_values[i] = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6

    return t_values, x_values, y_values


# Задані параметри
N = 18
a11 = 0.01 * N
a12 = 0.0001 * N
a21 = 0.0001 * N
a22 = 0.04 * N

# Початкові умови
x0 = 1000 - 10 * N  # Жертви
y0 = 700 - 10 * N  # Хижаки
h = 0.1
T = 150

# Розв'язання системи методом Рунге-Кутта
t_values, x_values, y_values = lotka_volterra_rk4(a11, a12, a21, a22, x0, y0, h, T)

# Візуалізація результатів
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_values, x_values, label="Жертви (x)")
plt.plot(t_values, y_values, label="Хижаки (y)")
plt.xlabel("Час (дні)")
plt.ylabel("Популяція")
plt.legend()
plt.title("Динаміка популяцій")

plt.subplot(1, 2, 2)
plt.plot(x_values, y_values, color='r')
plt.xlabel("Жертви (x)")
plt.ylabel("Хижаки (y)")
plt.title("Фазовий портрет")

plt.tight_layout()
plt.show()
