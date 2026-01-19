import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm

from dataset import MultiModalDataset
from model import MultimodalSentimentModel

# --- 配置 ---
BATCH_SIZE = 16
DATA_DIR = "../data"
VAL_FILE = "../data/val_split.csv" # 确保你之前跑 train.py 生成了这个文件
MODEL_PATH = "../output/best_model.pth"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def analyze():
    print(" 正在寻找 Bad Cases (预测错误的样本)...")
    
    # 准备数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = MultiModalDataset(DATA_DIR, VAL_FILE, transform=transform, mode='train')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultimodalSentimentModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    bad_cases = []
    
    with torch.no_grad():
        for texts, images, labels, guids in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            encoded = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids = encoded['input_ids'].to(device)
            mask = encoded['attention_mask'].to(device)
            
            outputs = model(input_ids, mask, images)
            _, preds = torch.max(outputs, 1)
            
            # 找出预测错误的
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    bad_cases.append({
                        'guid': guids[i],
                        'text': texts[i],
                        'true_label': idx_to_label[labels[i].item()],
                        'pred_label': idx_to_label[preds[i].item()]
                    })
                    
    # 保存错误记录
    df = pd.DataFrame(bad_cases)
    df.to_csv("../output/bad_cases.csv", index=False)
    print(f" 找到了 {len(df)} 个错误样本，已保存到 ../output/bad_cases.csv")
    
    # 打印前5个看看
    print("\n--- 错误样本示例 ---")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"GUID: {row['guid']}")
        print(f"文本: {row['text']}")
        print(f"真实: {row['true_label']} -> 预测: {row['pred_label']}")
        print("-" * 30)

if __name__ == "__main__":
    analyze()