import matplotlib.pyplot as plt

epochs = list(range(1, 26))
improved_accuracy = [ 0.3693, 0.5519, 0.4333,0.4577, 0.6452, 0.7578, 0.9021,0.6548,0.7090, 0.6971,  0.8731, 0.9492, 0.7910, 0.8820,  0.8359,0.9121,0.8241,0.8905
            , 0.8692,1.0000,0.9883, 0.9836,0.9620 ,  0.9620,1.0000]
simple_accuracy = [0.0800, 0.3879, 0.3882, 0.5538, 0.3185, 0.4583, 0.3799, 0.4736, 0.4694, 0.4779,
            0.5257, 0.3767, 0.5099, 0.3588, 0.4484, 0.3621, 0.3053, 0.4045, 0.4375, 0.4663,
            0.5855, 0.4732, 0.4696, 0.4155, 0.6220]

plt.figure(figsize=(12, 6))
plt.plot(epochs, improved_accuracy, label="Improved LSTM", marker="o", linestyle="-", color='b')
plt.plot(epochs, simple_accuracy, label="Simple LSTM", marker="o", linestyle="-", color='r')
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
