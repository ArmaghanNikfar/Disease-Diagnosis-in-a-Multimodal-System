import matplotlib.pyplot as plt

models = ['MLP', 'RNN', 'Random Forest', 'iLSTM', 'Simple LSTM']
accuracies = [0.7245, 0.8053, 0.6545, 0.8333, 0.8201]

colors = ['r', 'g', 'b', 'c', 'm']

plt.figure(figsize=(10, 6))

for i in range(len(models)):
    plt.plot([0, accuracies[i] * 100], [i, i], marker='o', linestyle='-', color=colors[i], label=f'{models[i]} ({accuracies[i]*100:.2f}%)')

plt.title('Model Accuracy Comparison', fontsize=16)
plt.xlabel('Accuracy (%)', fontsize=12)
plt.yticks(range(len(models)), models)  
plt.ylabel('Models', fontsize=12)

for i in range(len(models)):
    plt.text(accuracies[i] * 100 + 1, i, f'{accuracies[i]*100:.2f}%', va='center', fontsize=12)

plt.xlim(0, 100)
plt.grid(True)
plt.show()
