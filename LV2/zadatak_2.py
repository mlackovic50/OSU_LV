import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

# a)
print(f"Broj ispitanih osoba jest: {data.shape[0]}")

# b)
plt.scatter(data[:, 1], data[:, 2], color='blue', alpha=0.5)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

# c)
plt.scatter(data[::50, 1], data[::50, 2], color='red', alpha=0.5)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase (svaka 50. osoba)")
plt.show()

# d)
min_height = np.min(data[:, 1])
max_height = np.max(data[:, 1])
mean_height = np.mean(data[:, 1])

print(f"Minimalna visina: {min_height:.2f} cm")
print(f"Maksimalna visina: {max_height:.2f} cm")
print(f"Srednja visina: {mean_height:.2f} cm")

# e)
men = data[data[:, 0] == 1]
women = data[data[:, 0] == 0]

print(f"Muškarci - Min: {np.min(men[:, 1])}, Max: {np.max(men[:, 1])}, Srednja: {np.mean(men[:, 1])}")
print(f"Žene - Min: {np.min(women[:, 1])}, Max: {np.max(women[:, 1])}, Srednja: {np.mean(women[:, 1])}")

