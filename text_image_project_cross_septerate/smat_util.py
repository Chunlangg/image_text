from torchvision import models, transforms
from PIL import Image
import torch
'''

# Load ResNet model and remove the final classification layer
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()

# use transformer as resize method to resize all the image 

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_image_features(image_path):
    img = Image.open(image_path)
    
    # add another vector, decide the batch,(batch_size, channels, height, width)
    img_tensor = image_transform(img).unsqueeze(0)  
    with torch.no_grad():
        image_features = resnet_model(img_tensor)
        image_features=image_features.squeeze(1)
       #  print(f"the size of the image_feature: {image_features}")
    return image_features


# think about using the clip get the embeding 10/7 
'''
from torchvision import models, transforms
from PIL import Image
import torch
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel

# 加载 ResNet 模型并移除分类层
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()

# ResNet 图像预处理
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载 Vision Transformer (ViT) 模型和特征提取器
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# 定义投影层将 ViT 特征映射到 ResNet 的特征维度
vit_to_resnet_projection = nn.Linear(768, 2048)  # 将 ViT 的 768 维映射到 2048 维

def extract_image_features_resnet(image_path):
    """
    使用 ResNet 提取图像特征。
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = resnet_transform(img).unsqueeze(0)  # 添加 batch 维度
    resnet_model.eval()  # 设置为评估模式
    with torch.no_grad():
        image_features = resnet_model(img_tensor).squeeze(0)  # 移除 batch 维度
    return image_features

def extract_image_features_vit(image_path):
    """
    使用 Vision Transformer (ViT) 提取图像特征，并将其投影到与 ResNet 一致的维度。
    """
    img = Image.open(image_path).convert("RGB")
    inputs = vit_feature_extractor(images=img, return_tensors="pt")
    vit_model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs = vit_model(**inputs)
        vit_features = outputs.last_hidden_state[:, 0, :]  # 提取 [CLS] token 表示
    
    # 投影到 ResNet 的 2048 维
    vit_features_projected = vit_to_resnet_projection(vit_features).squeeze(0)
    return vit_features_projected
 


