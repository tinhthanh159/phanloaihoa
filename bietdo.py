import os
import matplotlib.pyplot as plt

data_dir = 'data/flowers/flowers'  # thư mục chứa 5 thư mục con là 5 lớp
classes = os.listdir(data_dir)
sample_counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in classes]

plt.figure(figsize=(8,6))
plt.bar(classes, sample_counts, color='skyblue')
plt.title("Số lượng ảnh theo từng lớp")
plt.xlabel("Loài hoa")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()
plt.show()
