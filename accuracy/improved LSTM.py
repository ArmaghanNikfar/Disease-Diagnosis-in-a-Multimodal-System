import matplotlib.pyplot as plt

epochs = list(range(1, 26))
accuracy = [ 0.3693, 0.5519, 0.4333,0.4577, 0.6452, 0.7578, 0.9021,0.6548,0.7090, 0.6971,  0.8731, 0.9492, 0.7910, 0.8820,  0.8359,0.9121,0.8241,0.8905
            , 0.8692,1.0000,0.9883, 0.9836,0.9620 ,  0.9620,1.0000]
loss = [1.33265, 0.9398, 4.25, 1.0776, 6.25, 0.9562, 0.8137,0.8759,0.7319, 0.6921, 0.5047, 0.5200,0.4418 , 0.6667,0.4282,0.3929, 0.4556,0.3127
        ,0.2536,0.1993,0.1374, 0.1791,0.2956, 0.1645,0.1890]


plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracy, label="Accuracy", marker="o", linestyle="-", color='b')
plt.plot(epochs, loss, label="Loss", marker="o", linestyle="-", color='r')

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Model: Accuracy & Loss over Epochs (Training vs. Validation)")
plt.legend()
plt.grid(True)
plt.show()
