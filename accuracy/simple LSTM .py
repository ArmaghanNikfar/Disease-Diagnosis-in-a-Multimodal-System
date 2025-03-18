import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [0.0800, 0.3879, 0.3882, 0.5538, 0.3185, 0.4583, 0.3799, 0.4736, 0.4694, 0.4779,
            0.5257, 0.3767, 0.5099, 0.3588, 0.4484, 0.3621, 0.3053, 0.4045, 0.4375, 0.4663,
            0.5855, 0.4732, 0.4696, 0.4155, 0.6220]
loss = [1.3875, 1.3825, 1.3781, 1.3645, 1.3751, 1.3617, 1.3613, 1.3324, 1.2957, 1.3114,
        1.0987, 1.2301, 1.0334, 1.2880, 1.1755, 1.2056, 1.2587, 0.9810, 1.0707, 1.0569,
        0.7231, 0.8371, 0.7895, 0.8425, 0.7755]


plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy, label=" Accuracy", marker="o", linestyle="-", color='b')
plt.plot(epochs, loss, label=" Loss", marker="o", linestyle="-", color='r')
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("RNN Model: Accuracy & Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
