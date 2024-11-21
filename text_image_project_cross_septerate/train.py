'''
import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dataset_loader import MultimodalDataset  # Your dataset class
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # Your models

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
    cross_validate(csv_file, k_folds=5, fusion_type="attention")

import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dataset_loader import MultimodalDataset  # Your dataset class
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # Your models


def set_seed(seed=42):
    """
    设置随机种子以保证实验可重复性。
    """
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


def train_model(train_loader, val_loader, class_weights, fusion_type="attention", device="cpu"):
    """
    训练模型并验证。
    """
    if fusion_type == "concat":
        print("Using simple concatenation for fusion.")
        model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=5).to(device)
    elif fusion_type == "attention":
        print("Using attention-based fusion.")
        model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=5).to(device)
    else:
        raise ValueError("Unknown fusion type. Choose 'concat' or 'attention'.")

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epoch_number):
        model.train()
        running_loss = 0.0
        for batch_idx, (img_data, text_data, labels) in enumerate(train_loader):
            img_data, text_data, labels = img_data.to(device), text_data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_data, text_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:  # 每10个batch打印一次
                print(f"Epoch [{epoch + 1}/{num_epoch_number}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每个 epoch 后验证模型
        avg_train_loss = running_loss / len(train_loader)
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
        
        logging.info(f"Epoch [{epoch + 1}/{num_epoch_number}], Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epoch_number}], Train Loss: {avg_train_loss:.4f}")
        print(f"Validation - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    return model


def evaluate_model(model, data_loader, device="cpu"):
    """
    模型评估。
    """
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for img_data, text_data, labels in data_loader:
            img_data, text_data, labels = img_data.to(device), text_data.to(device), labels.to(device)
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


def cross_validate(csv_file, k_folds=5, fusion_type="attention", device="cpu"):
    """
    k折交叉验证。
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
        model = train_model(train_loader, val_loader, class_weights, fusion_type, device)

        # 评估模型
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_validate(csv_file, k_folds=5, fusion_type="attention", device=device)
'''
import torch
import logging
import random
import numpy as np
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from dataset_loader import MultimodalDataset, load_data  # 数据集加载
from model import MultimodalClassifier, MultimodalClassifierWithAttention  # 模型定义


def set_seed(seed=42):
    """
    设置随机种子以保证实验可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 配置日志
logging.basicConfig(filename='training_log_seperte.log', level=logging.INFO)

# 超参数设置
num_epoch_number = 5
batch_size = 32


def train_model(train_loader, val_loader, class_weights, fusion_type="attention", device="cpu"):
    """
    训练模型并验证。
    """
    if fusion_type == "concat":
        print("Using simple concatenation for fusion.")
        model = MultimodalClassifier(image_dim=2048, text_dim=768, num_classes=5).to(device)
    elif fusion_type == "attention":
        print("Using attention-based fusion.")
        model = MultimodalClassifierWithAttention(image_dim=2048, text_dim=768, num_classes=5).to(device)
    else:
        raise ValueError("Unknown fusion type. Choose 'concat' or 'attention'.")

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(num_epoch_number):
        model.train()
        running_loss = 0.0
        for batch_idx, (img_data, text_data, labels) in enumerate(train_loader):
            img_data, text_data, labels = img_data.to(device), text_data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_data, text_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:  # 每10个batch打印一次
                print(f"Epoch [{epoch + 1}/{num_epoch_number}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 每个 epoch 后验证模型
        avg_train_loss = running_loss / len(train_loader)
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
        
        logging.info(f"Epoch [{epoch + 1}/{num_epoch_number}], Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epoch_number}], Train Loss: {avg_train_loss:.4f}")
        print(f"Validation - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    return model


def evaluate_model(model, data_loader, device="cpu"):
    """
    模型评估。
    """
    model.eval()
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for img_data, text_data, labels in data_loader:
            img_data, text_data, labels = img_data.to(device), text_data.to(device), labels.to(device)
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

    # # 可视化混淆矩阵
    # cm = confusion_matrix(all_labels, all_predictions)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()

    return accuracy, precision, recall, f1


def cross_validate(csv_file, k_folds=5, fusion_type="attention", image_embedding_type="cnn", device="cpu"):
    """
    k折交叉验证。
    """
    dataset = MultimodalDataset(csv_file, image_column="media_path", text_column="message_text", image_embedding_type=image_embedding_type)
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
        model = train_model(train_loader, val_loader, class_weights, fusion_type, device)

        # 评估模型
        accuracy, precision, recall, f1 = evaluate_model(model, val_loader, device)
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

    logging.info(f"For { fusion_type} with {image_embedding_type}: Cross-validation Results - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    print(f"Cross-validation Results - Accuracy: {avg_accuracy:.2f}%, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}")


if __name__ == "__main__":
    set_seed()
    csv_file = "output_final_data.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_validate(
        csv_file=csv_file,
        k_folds=3,
        fusion_type="attention",  # 可选 "attention"
        image_embedding_type="cnn",  # 可选 "cnn"
        device=device
    )





