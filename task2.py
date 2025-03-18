import numpy as np
import matplotlib.pyplot as plt


def epidemic_rk4(H, beta, gamma, x0, y0, z0, h, T):
    num_steps = int(T / h) + 1
    t_values = np.linspace(0, T, num_steps)
    x_values = np.zeros(num_steps)
    y_values = np.zeros(num_steps)
    z_values = np.zeros(num_steps)

    x_values[0] = x0
    y_values[0] = y0
    z_values[0] = z0

    def dx_dt(x, y):
        return -beta * x * y / H + 0.5 / gamma * y

    def dy_dt(x, y):
        return beta * x * y / H - 1 / gamma * y

    def dz_dt(y):
        return 0.5 / gamma * y

    for i in range(1, num_steps):
        x, y, z = x_values[i - 1], y_values[i - 1], z_values[i - 1]

        k1_x = h * dx_dt(x, y)
        k1_y = h * dy_dt(x, y)
        k1_z = h * dz_dt(y)

        k2_x = h * dx_dt(x + k1_x / 2, y + k1_y / 2)
        k2_y = h * dy_dt(x + k1_x / 2, y + k1_y / 2)
        k2_z = h * dz_dt(y + k1_y / 2)

        k3_x = h * dx_dt(x + k2_x / 2, y + k2_y / 2)
        k3_y = h * dy_dt(x + k2_x / 2, y + k2_y / 2)
        k3_z = h * dz_dt(y + k2_y / 2)

        k4_x = h * dx_dt(x + k3_x, y + k3_y)
        k4_y = h * dy_dt(x + k3_x, y + k3_y)
        k4_z = h * dz_dt(y + k3_y)

        x_values[i] = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        y_values[i] = y + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z_values[i] = z + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

    return t_values, x_values, y_values, z_values


# Задані параметри
N = 18
H = 1000 - N
beta = 25 - N
gamma = N

# Початкові умови
x0 = 900 - N  # Здорові
y0 = 90 - N  # Хворі
z0 = H - x0 - y0  # Одужалі
h = 0.1
T = 40

t_values, x_values, y_values, z_values = epidemic_rk4(H, beta, gamma, x0, y0, z0, h, T)

# Візуалізація результатів
plt.figure(figsize=(12, 5))
plt.plot(t_values, x_values, label="Здорові (x)")
plt.plot(t_values, y_values, label="Хворі (y)")
plt.plot(t_values, z_values, label="Одужалі (z)")
plt.xlabel("Час (дні)")
plt.ylabel("Кількість людей")
plt.legend()
plt.title("Динаміка розповсюдження епідемії")

plt.show()
