import torch
import torch.nn as nn
# from clip_image_embedding import extract_clip_image_features
# from clip_text_embedding import extract_clip_text_features

# Simple concatenation model
'''
class MultimodalClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes):
        super(MultimodalClassifier, self).__init__()
        self.fc_image = nn.Linear(image_dim, 256)
        self.fc_text = nn.Linear(text_dim, 256)

        # fusion image and text should be 256+256
        self.fc_fusion = nn.Linear(256*2, num_classes)
        #self.fc_fusion = nn.Linear(512, num_classes)
    
    # def forward(self, image_features=None, text_features=None):
    def forward(self, image_features, text_features):
        image_output = self.fc_image(image_features.squeeze(1))   # img_out 形状为 [batch_size, 256]
        text_output = self.fc_text(text_features.squeeze(1))

        # print(image_output.shape)  # 预期输出 [batch_size, 256]
        # print(text_output.shape)   # 预期输出 [batch_size, 256]
        
        combined_output = torch.cat((image_output, text_output), dim=1)
        output = self.fc_fusion(combined_output)
        return output


# Attention-based fusion model


class MultimodalClassifierWithAttention(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes):
        super(MultimodalClassifierWithAttention, self).__init__()
        
        # Linear layers for image and text features
        self.fc_image = nn.Linear(image_dim, 256)
        self.fc_text = nn.Linear(text_dim, 256)
        
        # Attention layer to calculate weights for image and text features
        self.attention_weights = nn.Linear(512, 2)  # Two weights: one for image, one for text
        
        # Final classification layer
        self.fc_fusion = nn.Linear(256, num_classes)
    
    def forward(self, image_features, text_features):
        # Process image and text features through fully connected layers

        image_output = torch.relu(self.fc_image(image_features))
        text_output = torch.relu(self.fc_text(text_features))

        # print(image_output.shape)  # 预期输出 [batch_size, 256]
        # print(text_output.shape)   # 预期输出 [batch_size, 256]

        image_output = self.fc_image(image_features.squeeze(1))   # img_out 形状为 [batch_size, 256]
        text_output = self.fc_text(text_features.squeeze(1))
        
        # Concatenate image and text features along the feature dimension
        combined_features = torch.cat((image_output, text_output), dim=1)  # Shape: [batch_size, 512]
        
        # Apply attention mechanism to weigh image and text features
        attention_scores = torch.softmax(self.attention_weights(combined_features), dim=1)
        image_weight, text_weight = attention_scores[:, 0], attention_scores[:, 1]
        
        # Expand weights to match the feature size
        image_weight = image_weight.unsqueeze(1)  # Shape: [batch_size, 1]
        text_weight = text_weight.unsqueeze(1)    # Shape: [batch_size, 1]
        
        # Apply the attention weights to image and text features
        fused_output = image_weight * image_output + text_weight * text_output  # Shape: [batch_size, 256]
        
        # Final classification layer
        output = self.fc_fusion(fused_output)
        
        return output
'''
import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes, image_embedding_type="cnn"):
        super(MultimodalClassifier, self).__init__()
        self.image_embedding_type = image_embedding_type
        self.fc_image = nn.Linear(image_dim, 256)
        self.fc_text = nn.Linear(text_dim, 256)
        self.fc_fusion = nn.Linear(256 * 2, num_classes)

    def forward(self, image_features, text_features):
        if self.image_embedding_type == "cnn":
            image_output = torch.relu(self.fc_image(image_features.view(image_features.size(0), -1)))
        elif self.image_embedding_type == "vit":
            image_output = torch.relu(self.fc_image(image_features.view(image_features.size(0), -1)))
        else:
            raise ValueError(f"Unsupported image_embedding_type: {self.image_embedding_type}")

        text_output = torch.relu(self.fc_text(text_features.view(text_features.size(0), -1)))
        combined_output = torch.cat((image_output, text_output), dim=1)
        output = self.fc_fusion(combined_output)
        return output


class MultimodalClassifierWithAttention(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes, image_embedding_type="cnn"):
        super(MultimodalClassifierWithAttention, self).__init__()
        self.image_embedding_type = image_embedding_type
        self.fc_image = nn.Linear(image_dim, 256)
        self.fc_text = nn.Linear(text_dim, 256)
        self.attention_layer = nn.Linear(512, 2)
        self.fc_fusion = nn.Linear(256, num_classes)

    def forward(self, image_features, text_features):
        if self.image_embedding_type == "cnn":
            image_output = torch.relu(self.fc_image(image_features.view(image_features.size(0), -1)))
        elif self.image_embedding_type == "vit":
            image_output = torch.relu(self.fc_image(image_features.view(image_features.size(0), -1)))
        else:
            raise ValueError(f"Unsupported image_embedding_type: {self.image_embedding_type}")

        text_output = torch.relu(self.fc_text(text_features.view(text_features.size(0), -1)))
        combined_features = torch.cat((image_output, text_output), dim=1)

        attention_scores = torch.softmax(self.attention_layer(combined_features), dim=1)
        image_weight, text_weight = attention_scores[:, 0].unsqueeze(1), attention_scores[:, 1].unsqueeze(1)
        fused_output = image_weight * image_output + text_weight * text_output

        output = self.fc_fusion(fused_output)
        return output
