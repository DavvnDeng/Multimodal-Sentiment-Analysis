import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, tokenizer=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.mode = mode
        
        # 读取 CSV
        # dtype={'guid': str} 强制把 guid 列当作字符串读取，防止变成 8.0
        try:
            self.df = pd.read_csv(label_file, dtype={'guid': str})
        except:
            self.df = pd.read_csv(label_file) # 如果没有表头或格式不对，回退默认
            
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- 核心修复逻辑 ---
        #不管读进来是 8 (int), "8" (str), 8.0 (float), "8.0" (str)，统统转成 "8"
        raw_guid = self.df.iloc[idx]['guid']
        try:
            # 先转 float 处理 8.0 的情况，再转 int 去掉小数，最后转 str
            guid = str(int(float(raw_guid)))
        except:
            # 如果是 weird string，就直接转 str
            guid = str(raw_guid)
        # ------------------
        
        # 1. 文本处理
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        content = ""
        try:
            with open(txt_path, 'r', encoding='gb18030', errors='ignore') as f:
                content = f.read().strip()
        except:
            content = "" # 读不到就空

        # 2. 图片处理
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # 黑图

        if self.transform:
            image = self.transform(image)

        # 3. 返回
        if self.mode == 'train':
            tag = self.df.iloc[idx]['tag']
            label = torch.tensor(self.label_map[tag], dtype=torch.long)
            return content, image, label, guid
        else:
            return content, image, guid