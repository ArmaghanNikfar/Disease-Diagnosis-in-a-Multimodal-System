import matplotlib.pyplot as plt


epochs = list(range(1, 26))
improved_LSTM_accuracy = [ 0.3693, 0.5519, 0.4333,0.4577, 0.6452, 0.7578, 0.9021,0.6548,0.7090, 0.6971,  0.8731, 0.9492, 0.7910, 0.8820,  0.8359,0.9121,0.8241,0.8905
            , 0.8692,1.0000,0.9883, 0.9836,0.9620 ,  0.9620,1.0000]

simple_accuracy = [0.0800, 0.3879, 0.3882, 0.5538, 0.3185, 0.4583, 0.3799, 0.4736, 0.4694, 0.4779,
            0.5257, 0.3767, 0.5099, 0.3588, 0.4484, 0.3621, 0.3053, 0.4045, 0.4375, 0.4663,
            0.5855, 0.4732, 0.4696, 0.4155, 0.6220]

Rnn_accuracy = [0.0275, 0.2660, 0.4199, 0.4906, 0.4070, 0.4842, 0.3527, 0.4136, 0.4615, 0.2326,
            0.3155, 0.5099, 0.4146, 0.3172, 0.3243, 0.3785, 0.4405, 0.6194, 0.4987, 0.5547,0.5784,0.5417,0.7602,0.6783,0.4006]

mlp_accuracy = [0.2955, 0.5000, 0.4697, 0.5511, 0.5095, 0.5814, 0.6837, 0.6326, 0.7254, 0.5303, 0.5095, 0.5814, 0.6326, 0.6023, 0.6023, 0.6534, 0.6326, 0.7140, 0.6837, 0.6534, 0.6837, 0.6837, 0.6629, 0.6629, 0.7140]

plt.figure(figsize=(12,6))
plt.plot(epochs, improved_LSTM_accuracy , label="iLSTM", marker="o", linestyle="-", color="r")
plt.plot(epochs, simple_accuracy , label='sLSTM', marker='o', linestyle='-', color='b')
plt.plot(epochs, Rnn_accuracy , label='RNN', marker='o', linestyle='-', color='g')
plt.plot(epochs, mlp_accuracy , label='MLP', marker='o', linestyle='-', color='m')
plt.xlabel("Epochs")
plt.ylabel("Models Accuracy ")
plt.legend()
plt.grid(True)
plt.show()