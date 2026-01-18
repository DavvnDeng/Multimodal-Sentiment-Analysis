import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
import pandas as pd
from PIL import Image

# 引入你的类
from dataset import MultiModalDataset
from model import MultimodalSentimentModel

# --- 配置 ---
DATA_DIR = "../data"
TEST_FILE = "../data/test_without_label.txt"
MODEL_PATH = "../output/best_model.pth"

def debug_one_sample():
    print("🕵️‍♂️ 开始深度侦探模式...")
    
    # 1. 模拟清洗过程，看看清洗后的 GUID 长啥样
    df = pd.read_csv(TEST_FILE)
    
    # 强制清洗逻辑
    def clean(val):
        try:
            return str(int(float(val))) # 8.0 -> 8
        except:
            return str(val)
    
    df['guid'] = df['guid'].apply(clean)
    
    # 取第一个样本的 GUID
    target_guid = df.iloc[0]['guid']
    print(f"\n👉 [检查 1] CSV读取的第一个 GUID 是: '{target_guid}' (类型: {type(target_guid)})")
    
    # 2. 检查文件是否真的存在
    img_path = os.path.join(DATA_DIR, f"{target_guid}.jpg")
    txt_path = os.path.join(DATA_DIR, f"{target_guid}.txt")
    
    print(f"\n👉 [检查 2] 正在去硬盘找文件: {img_path}")
    if os.path.exists(img_path):
        print("   ✅ 图片文件存在！")
        # 尝试读取图片
        try:
            img = Image.open(img_path).convert('RGB')
            print(f"   ✅ 图片读取成功，尺寸: {img.size}")
            # 检查是不是全黑
            extrema = img.getextrema()
            if extrema == ((0, 0), (0, 0), (0, 0)):
                print("   ⚠️ 警告：这是一张全黑图片！")
            else:
                print("   ✅ 图片看起来正常（不是全黑）。")
        except Exception as e:
            print(f"   ❌ 图片打开失败: {e}")
    else:
        print(f"   ❌ 致命错误：找不到图片文件！路径不对！")

    print(f"\n👉 [检查 3] 正在去硬盘找文本: {txt_path}")
    if os.path.exists(txt_path):
        print("   ✅ 文本文件存在！")
        # 尝试读取文本
        try:
            with open(txt_path, 'r', encoding='gb18030', errors='ignore') as f:
                content = f.read().strip()
            print(f"   ✅ 文本读取成功，内容预览: [{content[:50]}...]")
            if len(content) == 0:
                print("   ⚠️ 警告：文本是空的！")
        except Exception as e:
            print(f"   ❌ 文本打开失败: {e}")
    else:
        print(f"   ❌ 致命错误：找不到文本文件！")

    # 3. 让模型预测一次，看看概率
    print("\n👉 [检查 4] 模型预测诊断")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 加载模型
    model = MultimodalSentimentModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 构造一个单样本 Batch
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 强行读取
    with open(txt_path, 'r', encoding='gb18030', errors='ignore') as f:
        text_real = f.read().strip()
    img_real = Image.open(img_path).convert('RGB')
    
    # 预处理
    img_tensor = transform(img_real).unsqueeze(0).to(device)
    encoded = tokenizer([text_real], return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    input_ids = encoded['input_ids'].to(device)
    mask = encoded['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        logits = model(input_ids, mask, img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print(f"   模型原始输出 (Logits): {logits.cpu().numpy()}")
    print(f"   模型概率分布 (Probs):  {probs.cpu().numpy()}")
    print(f"   最终预测: {idx_to_label[pred_idx]}")
    
    if probs[0][0] > 0.9:
        print("\n😱 诊断结果：模型极其确信这是 Negative。如果所有样本都这样，说明模型可能过拟合了，或者输入数据在 Dataset 类内部被截断了。")
    else:
        print("\n🤔 诊断结果：概率分布看起来还算正常，如果只有这一个是 Negative 可能是巧合。")

if __name__ == "__main__":
    debug_one_sample()