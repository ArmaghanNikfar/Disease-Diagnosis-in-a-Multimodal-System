import matplotlib.pyplot as plt

iLSTM = "1.0000"
sLSTM = "0.8333"
RNN = "0.8333"
MLP = "0.6667"
Random_Forest = "0.4364"

models = ['iLSTM', 'sLSTM', 'RNN', 'MLP', 'Random_Forest']
accuracies = [1.0000, 0.8333, 0.8333, 0.6667, 0.4364]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies , color=['r', 'b', 'g', 'm', 'c'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)  
plt.show()