import matplotlib.pyplot as plt

iLSTM = "0.9667"
sLSTM = "0.9333"
RNN = "0.8333"
MLP = "0.9000"
Random_Forest = "0.4364"

models = ['iLSTM', 'sLSTM', 'RNN', 'MLP', 'Random_Forest']
accuracies = [0.9667, 0.9333, 0.8333, 0.9000, 0.4364]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies , color=['r', 'b', 'g', 'm', 'c'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)  
plt.show()