import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [ 0.4018,0.5049,0.6713,0.7053,0.7698,0.8154,0.8659,0.8031,0.8744,0.8936,0.9312,0.8728,0.9031,0.9073,0.8489,0.9333,0.9168,0.9731,
            0.9018, 0.9458,0.9921,0.9649,0.9581, 0.9290,0.9667 ]
loss = [1.2219,0.9898,0.8675,0.6835,0.5332,0.4974,0.4218,0.4003 ,0.3631,0.2846,0.3493,0.2416,0.3157,0.1935,0.2327,0.3416,0.1482,0.2008,
        0.0868,0.2543,0.1702,0.0780,0.0919,0.1399,0.0755]


plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy, label="Accuracy", marker="o", linestyle="-")
plt.plot(epochs, loss, label="Loss", marker="o", linestyle="-")

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Improved LSTM")
plt.legend()
plt.grid(True)
plt.show()
