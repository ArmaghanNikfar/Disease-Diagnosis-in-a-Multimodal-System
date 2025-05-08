import matplotlib.pyplot as plt

# داده‌های دقت و خطا از خروجی مدل
epochs = list(range(1, 26))


mlp_accuracy = [0.2630,0.5189, 0.5548,0.5357, 0.6583,0.7438 ,0.8230 ,0.8743 , 0.8666,0.8675,0.9121 ,0.8558 ,0.8971, 0.8145, 0.9215 ,0.9381,
                0.9200,0.9368 ,0.9386,0.9846 ,0.9419 , 0.9459 ,0.9900 ,0.9484 ,0.9000]
mlp_loss =     [1.3602,1.2167 ,1.0689 ,0.9930,0.8189,0.7380,0.6076,0.5277,0.4548,0.3627,0.3272,0.3934 ,0.2778,0.3914,0.2156,0.2016,0.1974, 0.1607,0.1654,
                 0.1325,0.1634 ,0.1766,0.1123 ,0.1492,0.2686]

# رسم نمودار
plt.figure(figsize=(10, 5))
plt.plot(epochs, mlp_accuracy, label="Accuracy", marker="o", linestyle="-")
plt.plot(epochs, mlp_loss, label="Loss", marker="o", linestyle="-")

# تنظیمات نمودار
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("MLP model")
plt.legend()
plt.grid(True)

# نمایش نمودار
plt.show()
