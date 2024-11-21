from torchvision import models, transforms
from PIL import Image
import torch

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

