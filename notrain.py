import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import gradio as gr
from PIL import Image

# Các class label tương ứng (5 loại hoa trong bộ dataset)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Chuẩn bị transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load model ResNet-50 đã fine-tune (nếu có file .pth, bạn load state_dict ở đây)
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
# resnet.load_state_dict(torch.load('resnet_flowers.pth'))
resnet.eval()

# Load model EfficientNet-B0 đã fine-tune (nếu có file .pth, bạn load state_dict ở đây)
effnet = models.efficientnet_b0(pretrained=True)
effnet.classifier[1] = nn.Linear(effnet.classifier[1].in_features, len(class_names))
# effnet.load_state_dict(torch.load('effnet_flowers.pth'))
effnet.eval()

def predict(image, model_name):
    image = Image.fromarray(image).convert('RGB')
    x = transform(image).unsqueeze(0)
    if model_name == 'resnet':
        outputs = resnet(x)
    else:
        outputs = effnet(x)
    probs = torch.softmax(outputs, dim=1).detach().numpy()[0]
    max_idx = probs.argmax()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Dự đoán loại hoa với ResNet-50 và EfficientNet-B0")
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Tải ảnh lên")
        model_selector = gr.Dropdown(choices=['resnet', 'efficientnet'], value='resnet', label="Chọn mô hình")
    output = gr.Label(num_top_classes=5, label="Dự đoán xác suất các lớp")
    btn = gr.Button("Dự đoán")

    btn.click(fn=predict, inputs=[image_input, model_selector], outputs=output)

if __name__ == "__main__":
    demo.launch()