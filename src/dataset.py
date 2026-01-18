import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.df = pd.read_csv(label_file)
        # 标签映射
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        guid = str(self.df.iloc[idx]['guid'])
        
        # 1. 文本处理 (尝试不同编码读取)
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        content = ""
        try:
            with open(txt_path, 'r', encoding='gb18030', errors='ignore') as f:
                content = f.read().strip()
        except:
            pass # 读不到就空着

        # 2. 图片处理
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # 坏图补黑

        if self.transform:
            image = self.transform(image)

        # 3. 返回结果
        if self.mode == 'train':
            tag = self.df.iloc[idx]['tag']
            label = torch.tensor(self.label_map[tag], dtype=torch.long)
            return content, image, label
        else:
            return content, image, guid

# 测试代码
if __name__ == "__main__":
    print("正在测试 Dataset...")
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    # 注意：这里路径要根据你在哪运行来定，如果在src下运行，data在上一级
    ds = MultiModalDataset("../data", "../data/train.txt", transform=transform)
    print(f"读取成功，样本数：{len(ds)}")
    print(f"第一条数据：{ds[0]}")