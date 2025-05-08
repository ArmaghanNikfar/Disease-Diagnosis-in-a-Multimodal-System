import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [0.3070,0.4492,0.2664,0.3814,0.5418,0.6859 ,0.7139,0.6618,0.5574,0.6755,0.7462,0.7510 ,0.6820, 0.7632,00.8521,0.7432,0.7430,0.7820,
            0.8197,0.7983,0.8244,0.8853, 0.8395 , 0.7807, 0.9333]
loss = [1.3639, 1.2505,1.2190,1.0704,0.9233,0.8605,0.8896,0.8548,0.8134,0.8562 ,0.6250,0.7469,0.6681,0.5921,0.5921 ,0.6327,0.6260 ,
        0.5757,0.5836,0.5411,0.4553, 0.3336,0.4312,0.5693, 0.3039]


plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy, label=" Accuracy", marker="o", linestyle="-")
plt.plot(epochs, loss, label=" Loss", marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Simple LSTM")
plt.legend()
plt.grid(True)
plt.show()
