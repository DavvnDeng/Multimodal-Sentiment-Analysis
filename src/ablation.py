import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm

from dataset import MultiModalDataset
from model import MultimodalSentimentModel

# --- 配置 ---
BATCH_SIZE = 16
MAX_LEN = 128
DATA_DIR = "../data"
# 使用之前划分好的验证集
VAL_FILE = "../data/val_split.csv" 
MODEL_PATH = "../output/best_model.pth"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def evaluate(model, loader, tokenizer, mode='multimodal'):
    """
    mode: 
      - 'multimodal': 正常预测
      - 'text_only': 把图片全置为0
      - 'image_only': 把文本全置为0
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, images, labels, guids in tqdm(loader, desc=f"Testing {mode}"):
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
            
            # === 核心：消融逻辑 ===
            if mode == 'text_only':
                # 将图片全设为 0 (黑色)
                images = torch.zeros_like(images)
            elif mode == 'image_only':
                # 将文本 Mask 全设为 0 (让模型以为没有任何字)
                attention_mask = torch.zeros_like(attention_mask)
                input_ids = torch.zeros_like(input_ids)
            # ===================
            
            outputs = model(input_ids, attention_mask, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    return acc

def run_ablation():
    print("🔬 开始消融实验...")
    
    # 准备数据和模型
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = MultiModalDataset(DATA_DIR, VAL_FILE, transform=transform, mode='train')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultimodalSentimentModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 1. 测试多模态 (完整)
    acc_multi = evaluate(model, val_loader, tokenizer, mode='multimodal')
    print(f" 多模态 (Multimodal) 准确率: {acc_multi:.2f}%")
    
    # 2. 测试仅文本
    acc_text = evaluate(model, val_loader, tokenizer, mode='text_only')
    print(f" 仅文本 (Text Only) 准确率: {acc_text:.2f}%")
    
    # 3. 测试仅图像
    acc_img = evaluate(model, val_loader, tokenizer, mode='image_only')
    print(f" 仅图像 (Image Only) 准确率: {acc_img:.2f}%")
    
    print("\n--- 实验结论建议 ---")
    print(f"你的报告里应该画个表，填入这三个数字。")
    print("理论上，多模态应该最高，其次是文本，图片通常最低。")

if __name__ == "__main__":
    run_ablation()
