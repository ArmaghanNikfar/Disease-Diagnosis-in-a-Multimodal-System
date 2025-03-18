import matplotlib.pyplot as plt

# داده‌های دقت و خطا از خروجی مدل
epochs = list(range(1, 27))


mlp_accuracy = [0.2955, 0.5000, 0.4697, 0.5511, 0.5095, 0.5814, 0.6837, 0.6326, 0.7254, 0.5303, 0.5095, 0.5814, 0.6326, 0.6023, 0.6023, 0.6534, 0.6326, 0.7140, 0.6837, 0.6534, 0.6837, 0.6837, 0.6629, 0.6629, 0.7140, 0.7860]
mlp_loss =     [1.4155, 1.3096, 1.2838, 1.2470, 1.2489, 1.2180, 1.1752, 1.1170, 1.0313, 1.1322, 1.1000, 1.0714, 0.9714, 0.9658, 0.9526, 0.8603, 0.8890, 0.8386, 0.7752, 0.7641, 0.7487, 0.7249, 0.7087, 0.6969, 0.6262, 0.5918]

# رسم نمودار
plt.figure(figsize=(10, 5))
plt.plot(epochs, mlp_accuracy, label="Accuracy", marker="o", linestyle="-")
plt.plot(epochs, mlp_loss, label="Loss", marker="o", linestyle="-")

# تنظیمات نمودار
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Accuracy & Loss over Epochs")
plt.legend()
plt.grid(True)

# نمایش نمودار
plt.show()
