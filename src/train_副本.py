import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import time

# 引入我们之前写的类
from dataset import MultiModalDataset
from model import MultimodalSentimentModel

# --- 1. 超参数设置 (根据PPT要求调整这里) ---
BATCH_SIZE = 16          # 如果显存(内存)不够报错，改小一点，比如 8
LEARNING_RATE = 2e-5     # 学习率，BERT微调通常用很小的数
EPOCHS = 5               # 训练轮数，通常3-5轮就收敛了
MAX_LEN = 128            # 文本最长长度
DATA_DIR = "../data"     # 数据文件夹路径
TRAIN_FILE = "../data/train.txt"
OUTPUT_DIR = "../output" # 模型保存路径

# 自动检测设备：Mac优先用MPS，Nvidia显卡用CUDA，否则用CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("使用 MPS (Mac GPU) 加速训练")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("使用 CUDA (Nvidia GPU) 加速训练")
else:
    DEVICE = torch.device("cpu")
    print("未检测到GPU，使用 CPU 训练 (会比较慢)")

def train():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 数据准备与划分验证集 ---
    print("正在划分训练集和验证集...")
    # 读取原始 train.txt
    df = pd.read_csv(TRAIN_FILE)
    
    # 划分 80% 训练, 20% 验证 (random_state保证结果可复现)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 保存临时文件给Dataset读取
    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)
    
    print(f"训练集数量: {len(train_df)}, 验证集数量: {len(val_df)}")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化 Dataset
    train_dataset = MultiModalDataset(DATA_DIR, os.path.join(DATA_DIR, "train_split.csv"), transform=transform)
    val_dataset = MultiModalDataset(DATA_DIR, os.path.join(DATA_DIR, "val_split.csv"), transform=transform)

    # 实例化 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 初始化模型和工具 ---
    print("正在加载模型和Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultimodalSentimentModel(num_classes=3).to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    # --- 4. 开始训练循环 ---
    for epoch in range(EPOCHS):
        print(f"\n======== Epoch {epoch + 1} / {EPOCHS} ========")
        
        # === 训练阶段 ===
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc="Training")
        
        for texts, images, labels, guids in progress_bar:
            # 数据移到设备
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 文本 Tokenize (分词)
            encoded_text = tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            
            input_ids = encoded_text['input_ids'].to(DEVICE)
            attention_mask = encoded_text['attention_mask'].to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask, images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})

        train_acc = 100 * correct / total
        print(f"Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # === 验证阶段 ===
        model.eval()
        val_correct = 0
        val_total = 0
        
        print("Running Validation...")
        with torch.no_grad():
            for texts, images, labels, guids in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                encoded_text = tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors='pt'
                )
                input_ids = encoded_text['input_ids'].to(DEVICE)
                attention_mask = encoded_text['attention_mask'].to(DEVICE)
                
                outputs = model(input_ids, attention_mask, images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Validation Acc: {val_acc:.2f}%")

        # === 保存最佳模型 ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"新的最佳模型已保存 (Acc: {val_acc:.2f}%)")

    print("\n训练完成！最佳验证集准确率: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    train()