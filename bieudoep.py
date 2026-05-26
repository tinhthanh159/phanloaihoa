import matplotlib.pyplot as plt
import numpy as np

# --- Dữ liệu huấn luyện và kiểm tra cho ResNet-50 (lấy từ ảnh log của bạn) ---
# train_accuracy_resnet: Độ chính xác trên tập huấn luyện của ResNet-50 qua 5 epoch
train_accuracy_resnet = [82.18, 95.86, 98.61, 99.33, 98.93] # Đơn vị %
# train_loss_resnet: Hàm mất mát trên tập huấn luyện của ResNet-50 qua 5 epoch
train_loss_resnet = [0.5815, 0.1360, 0.0547, 0.0289, 0.0387]

# test_accuracy_resnet: Độ chính xác cuối cùng trên tập test là 93.99%.
# Để vẽ đồ thị, chúng ta giả định giá trị này ổn định qua các epoch nếu không có dữ liệu từng epoch.
test_accuracy_resnet = [93.99] * 5
# test_loss_resnet: Hàm mất mát trên tập test không được hiển thị trực tiếp theo từng epoch trong log.
# Dựa trên hình ảnh bạn cung cấp cho ResNet-50, có một đường test loss màu đỏ gạch đứt
# có vẻ ổn định ở mức ~0.2. Tôi sẽ lấy giá trị này.
test_loss_resnet = [0.20] * 5 # Giá trị lấy từ đồ thị bạn cung cấp (~0.2)


# --- Dữ liệu huấn luyện và kiểm tra cho EfficientNet-B0 (lấy từ ảnh log của bạn) ---
# train_accuracy_effnet: Độ chính xác trên tập huấn luyện của EfficientNet-B0 qua 5 epoch
train_accuracy_effnet = [77.95, 92.27, 95.89, 97.89, 98.09] # Đơn vị %
# train_loss_effnet: Hàm mất mát trên tập huấn luyện của EfficientNet-B0 qua 5 epoch
train_loss_effnet = [0.7717, 0.2527, 0.1396, 0.0774, 0.0656]

# test_accuracy_effnet: Độ chính xác cuối cùng trên tập test là 93.76%.
test_accuracy_effnet = [93.76] * 5
# test_loss_effnet: Hàm mất mát (loss) trên tập test không được hiển thị trực tiếp theo từng epoch trong log.
# Tương tự ResNet-50, tôi sẽ giả định một giá trị ổn định cho EfficientNet-B0.
# Dựa vào kết quả tổng thể và so sánh, có thể giả định nó quanh mức 0.25-0.3.
test_loss_effnet = [0.25] * 5 # Giá trị giả định, bạn có thể điều chỉnh nếu có dữ liệu chính xác hơn


# --- Hàm để vẽ biểu đồ ---
def plot_training_history(train_acc, test_acc, train_loss, test_loss, model_name):
    """
    Vẽ biểu đồ độ chính xác và hàm mất mát của quá trình huấn luyện.

    Args:
        train_acc (list): Danh sách độ chính xác trên tập huấn luyện qua các epoch.
        test_acc (list): Danh sách độ chính xác trên tập kiểm tra qua các epoch.
        train_loss (list): Danh sách hàm mất mát trên tập huấn luyện qua các epoch.
        test_loss (list): Danh sách hàm mất mát trên tập kiểm tra qua các epoch.
        model_name (str): Tên của mô hình để hiển thị trên tiêu đề biểu đồ.
    """
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5)) # Kích thước tổng thể của figure

    # Subplot 1: Biểu đồ Độ chính xác
    plt.subplot(1, 2, 1) # 1 hàng, 2 cột, vị trí 1
    plt.plot(epochs, train_acc, 'b-', label='Độ chính xác huấn luyện') # 'b-' là màu xanh, đường liền nét
    plt.plot(epochs, test_acc, 'r--', label='Độ chính xác kiểm tra') # 'r--' là màu đỏ, đường gạch đứt
    plt.title(f'Biểu đồ độ chính xác của {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác (%)')
    plt.legend() # Hiển thị chú thích (label)
    plt.grid(True) # Hiển thị lưới
    plt.ylim(min(min(train_acc), min(test_acc)) * 0.95, max(max(train_acc), max(test_acc)) * 1.05) # Đảm bảo trục Y vừa đủ

    # Subplot 2: Biểu đồ Hàm mất mát
    plt.subplot(1, 2, 2) # 1 hàng, 2 cột, vị trí 2
    plt.plot(epochs, train_loss, 'b-', label='Hàm mất mát huấn luyện')
    plt.plot(epochs, test_loss, 'r--', label='Hàm mất mát kiểm tra')
    plt.title(f'Biểu đồ hàm mất mát của {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Hàm mất mát')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(min(train_loss), min(test_loss)) * 0.9, max(max(train_loss), max(test_loss)) * 1.1) # Đảm bảo trục Y vừa đủ


    plt.tight_layout() # Điều chỉnh khoảng cách giữa các subplot để tránh chồng lấn
    plt.show() # Hiển thị các biểu đồ

# --- Gọi hàm để vẽ biểu đồ cho từng mô hình ---

print("Vẽ biểu đồ cho ResNet-50...")
plot_training_history(train_accuracy_resnet, test_accuracy_resnet,
                      train_loss_resnet, test_loss_resnet, "ResNet-50")

print("\nVẽ biểu đồ cho EfficientNet-B0...")
plot_training_history(train_accuracy_effnet, test_accuracy_effnet,
                      train_loss_effnet, test_loss_effnet, "EfficientNet-B0")