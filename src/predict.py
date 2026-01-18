import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd

from dataset import MultiModalDataset
from model import MultimodalSentimentModel

# --- 配置 ---
BATCH_SIZE = 16
MAX_LEN = 128
DATA_DIR = "../data"
TEST_FILE = "../data/test_without_label.txt" # 直接读原始文件！
MODEL_PATH = "../output/best_model.pth"
OUTPUT_FILE = "../test_result.txt"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def predict():
    print(" 开始最终预测...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 直接加载，依靠 Dataset 内部的修复逻辑
    test_dataset = MultiModalDataset(DATA_DIR, TEST_FILE, transform=transform, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MultimodalSentimentModel(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    results = []
    
    with torch.no_grad():
        for texts, images, guids in tqdm(test_loader):
            images = images.to(DEVICE)
            encoded_text = tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
            input_ids = encoded_text['input_ids'].to(DEVICE)
            attention_mask = encoded_text['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, images)
            _, preds = torch.max(outputs, 1)
            
            for guid, pred_idx in zip(guids, preds.cpu().numpy()):
                results.append({'guid': guid, 'tag': idx_to_label[pred_idx]})
                
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n 预测完成！类别分布如下：")
    print(df['tag'].value_counts())

if __name__ == "__main__":
    predict()