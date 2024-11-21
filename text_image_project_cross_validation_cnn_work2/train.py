"""
import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset_loader import load_data  # Load the dataset with train/test split
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # Import both models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for GPU operations.
    torch.backends.cudnn.benchmark = False  # Disable benchmark
logging.basicConfig(filename='training_log.log', level=logging.INFO)
#logging.basicConfig(filename='new_training_log.log', level=logging.INFO)
num_epoch_number=2
epoch_list=[]
accuracy_list=[]
precision_list=[]
recall_list=[]
f1_score_list=[]



def train_model(fusion_type="attention"):
    '''
    Train either the simple concatenation model or the attention-based model
    Args:
    - fusion_type (str): "concat" for simple concatenation, "attention" for attention-based fusion.
    '''
    # Load train and test data
    train_loader, test_loader = load_data('output_final_data.csv')
    # set logging
  
    # Initialize the model based on the selected fusion type
    if fusion_type == "concat":
        print("Using simple concatenation for fusion.")
        model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=5)
    elif fusion_type == "attention":
        print("Using attention-based fusion.")
        model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=5)
    
    else:
        raise ValueError("Unknown fusion type. Choose 'concat' or 'attention'.")

    # Initialize the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss expects labels 0-5
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = num_epoch_number
    for epoch in range(num_epochs):
        model.train()
        for img_data, text_data, labels in train_loader:
            optimizer.zero_grad()
            # Forward pass
            outputs = model(img_data, text_data)
            # print(f"Outputs shape: {outputs.shape}")  # should be [32, 5]
            # print(f"Labels shape: {labels.shape}")    # should be [32]
            # print(f"Labels: {labels}")                # make sure the label is int

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # log important information
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    #Evaluate the model
    accuracy,precision,recall,f1= evaluate_model(model, test_loader)
    logging.info(f"For {num_epochs} epoches, the accuracy is: {accuracy:.4f}, the precision is:{precision:.4f}, the recall is{recall:.4f}, the f1 score is: {f1:.4f}")
    logging.info(f" ")

    # 新加11/20/9:47
    return model
def evaluate_model(model, test_loader):
    '''
    Evaluate the model on the test set and print the accuracy.
    '''
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for img_data, text_data, labels in test_loader:
            outputs = model(img_data, text_data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())  # Move to CPU and convert to numpy
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total

      # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)
       
       
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f' For {num_epoch_number} Test Accuracy: {accuracy:.2f}%')

    return accuracy,precision,recall,f1
    

if __name__ == "__main__":
    # You can switch between 'concat' and 'attention' for comparison
    fusion_type = "attention"  # Change this to 'attention' for the attention-based model
    logging.info(f"Training with {fusion_type} fusion")
    # train_model(fusion_type)

    # 新加11/20/9:47 
    model = train_model(fusion_type)
    # Optionally evaluate again at the end to ensure consistency

    # 新加 11/20/9:47
    # logging.info(f"Final Evaluation with {fusion_type} fusion")
    # test_loader = load_data('output_final_data.csv')[1]  # Reload test_loader if necessary
    # evaluate_model(model, test_loader)
"""

import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dataset_loader import MultimodalDataset  # dataset 
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # multimodal

# 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO)

# 超参数设置
num_epoch_number = 5
batch_size = 32

# 模型训练
def train_model(train_loader, val_loader, class_weights, fusion_type="attention"):
    """
    Train either the simple concatenation model or the attention-based model.
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        class_weights: Weight tensor for CrossEntropyLoss.
        fusion_type (str): "concat" for simple concatenation, "attention" for attention-based fusion.
    Returns:
        Trained model.
    """
    if fusion_type == "concat":
        print("Using simple concatenation for fusion.")
        model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=5)
    elif fusion_type == "attention":
        print("Using attention-based fusion.")
        model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=5)
    else:
        raise ValueError("Unknown fusion type. Choose 'concat' or 'attention'.")

    # 使用加权损失函数
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epoch_number):
        model.train()
        for img_data, text_data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(img_data, text_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 每个 epoch 后验证
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epoch_number}], Loss: {loss.item():.4f}")
        logging.info(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epoch_number}], Loss: {loss.item():.4f}")

    return model

# 模型评估
def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given DataLoader and return metrics.
    """
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for img_data, text_data, labels in data_loader:
            outputs = model(img_data, text_data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算评估指标
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=1)

    # 输出分类报告
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, zero_division=1))

    # 可视化混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    return accuracy, precision, recall, f1

# 交叉验证
def cross_validate(csv_file, k_folds=5, fusion_type="attention"):
    """
    Perform k-fold cross-validation.
    Args:
        csv_file: Path to the CSV file.
        k_folds (int): Number of folds for cross-validation.
        fusion_type (str): "concat" or "attention".
    """
    dataset = MultimodalDataset(csv_file, image_column="media_path", text_column="message_text")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 统计类别分布
    class_counts = dataset.data['Category'].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)

    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.data['Category'])):
        print(f"Fold {fold + 1}/{k_folds}")
        logging.info(f"Fold {fold + 1}/{k_folds}")

        # 创建训练和验证集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 训练并验证
        model = train_model(train_loader, val_loader, class_weights, fusion_type)

        # 评估模型
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)

        logging.info(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        print()

    # 汇总结果
    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_precision = np.mean(fold_results['precision'])
    avg_recall = np.mean(fold_results['recall'])
    avg_f1 = np.mean(fold_results['f1'])

    logging.info(f"Cross-validation Results - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print(f"Cross-validation Results - Accuracy: {avg_accuracy:.2f}%, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")

if __name__ == "__main__":
    set_seed()
    csv_file = "output_final_data.csv"
    cross_validate(csv_file, k_folds=5, fusion_type="concat")




"""


# newest
import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dataset_loader import load_data, MultimodalDataset  # Load the dataset with train/test split
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # Import both models

# Seed setting for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO)
num_epoch_number = 5
batch_size = 32
accuracy_list, precision_list, recall_list, f1_score_list = [], [], [], []

def train_model(train_loader, val_loader, fusion_type="attention"):
    '''
    Train either the simple concatenation model or the attention-based model.
    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        fusion_type (str): "concat" for simple concatenation, "attention" for attention-based fusion.
    Returns:
        Trained model.
    '''
    # Initialize the model based on the selected fusion type
    if fusion_type == "concat":
        print("Using simple concatenation for fusion.")
        model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=5)
    elif fusion_type == "attention":
        print("Using attention-based fusion.")
        model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=5)
    else:
        raise ValueError("Unknown fusion type. Choose 'concat' or 'attention'.")

    # Initialize the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epoch_number):
        model.train()
        for img_data, text_data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(img_data, text_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set after each epoch
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epoch_number}], Loss: {loss.item():.4f}")
        logging.info(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epoch_number}], Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, data_loader):
    '''
    Evaluate the model on the given DataLoader and return metrics.
    '''
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for img_data, text_data, labels in data_loader:
            outputs = model(img_data, text_data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

def cross_validate(csv_file, k_folds=5, fusion_type="attention"):
    '''
    Perform k-fold cross-validation.
    Args:
        csv_file: Path to the CSV file.
        k_folds (int): Number of folds for cross-validation.
        fusion_type (str): "concat" or "attention".
    '''
    # Load dataset
    dataset = MultimodalDataset(csv_file, image_column="media_path", text_column="message_text")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset.data['Category'])):
        print(f"Fold {fold + 1}/{k_folds}")
        logging.info(f"Fold {fold + 1}/{k_folds}")

        # Create train and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train and validate
        model = train_model(train_loader, val_loader, fusion_type)

        # Evaluate on validation set
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader)
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)

        logging.info(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    # Aggregate results
    avg_accuracy = np.mean(fold_results['accuracy'])
    avg_precision = np.mean(fold_results['precision'])
    avg_recall = np.mean(fold_results['recall'])
    avg_f1 = np.mean(fold_results['f1'])

    logging.info(f"Cross-validation Results - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print(f"Cross-validation Results - Accuracy: {avg_accuracy:.2f}%, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")

if __name__ == "__main__":
    set_seed()
    csv_file = "output_final_data.csv"
    cross_validate(csv_file, k_folds=5, fusion_type="concat")
"""
