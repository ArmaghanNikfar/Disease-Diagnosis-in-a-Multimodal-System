import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [0.0275, 0.2660, 0.4199, 0.4906, 0.4070, 0.4842, 0.3527, 0.4136, 0.4615, 0.2326,
            0.3155, 0.5099, 0.4146, 0.3172, 0.3243, 0.3785, 0.4405, 0.6194, 0.4987, 0.5547,0.5784,0.5417,0.7602,0.6783,0.4006]
loss = [1.4473, 1.3874, 1.2727, 1.2553, 1.2115, 1.2671, 1.2338, 1.3127, 1.1100, 1.3681,
        1.2867, 1.1233, 1.1419, 1.2498, 1.1960, 1.1384, 1.2061, 0.9088, 1.1366, 1.0448,1.0322,1.1190,0.7598,0.9911,1.0384]

plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy, label=" Accuracy", marker="o", linestyle="-")
plt.plot(epochs, loss, label=" Loss", marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("RNN Model: Accuracy & Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
