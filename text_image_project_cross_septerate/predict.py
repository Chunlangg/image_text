# import torch
# from model import  MultimodalClassifier,MultimodalClassifierWithAttention  # Change if using another model
# from dataset_loader import load_data



# def predict(fusion_type="concat"):
#     _, test_loader = load_data('your_data.csv', test_split=0.2)

#     # Load the model (adjust to use the correct fusion type)
#     if fusion_type == "concat":
#         model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=6)
#         model.load_state_dict(torch.load('trained_model_concat.pth'))
#     elif fusion_type == "attention":
#         model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=6)
#         model.load_state_dict(torch.load('trained_model_attention.pth'))
#     else:
#         raise ValueError("Unknown fusion type.")

#     model.eval()
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for img_data, text_data, labels in test_loader:
#             outputs = model(img_data, text_data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f'Test Accuracy: {accuracy:.2f}%')

# if __name__ == "__main__":
#     fusion_type = "concat"  # or "attention"
#     predict(fusion_type)
