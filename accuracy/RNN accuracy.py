import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [ 0.2587, 0.3293 , 0.3479, 0.4002, 0.3880, 0.4009,0.3495, 0.4378 , 0.4984, 0.4193,
            0.5500, 0.5938,  0.5721, 0.5668, 0.5692,0.6140,  0.6036, 0.5773, 0.6312, 0.6315,0.5964,0.5992, 0.6943,0.6783,0.8333]
loss = [1.3946, 1.3320, 1.2644, 1.2481, 1.2246, 1.1990, 1.1913, 1.1457, 1.0383, 1.0490,
        0.9978,  0.9351, 0.9214,0.9134, 0.8612, 0.7781, 0.7801, 0.7482, 0.7679, 0.7413,0.7850,0.7835,0.7125,0.6616,0.5126]

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy, label=" Accuracy", marker="o", linestyle="-")
plt.plot(epochs, loss, label=" Loss", marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("RNN Model")
plt.legend()
plt.grid(True)
plt.show()
