import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=False):
        """
        Args:
            num_classes: 分类数量 (negative, neutral, positive -> 3)
            freeze_backbone: 是否冻结预训练模型的参数 (显存不够时设为True)
        """
        super(MultimodalSentimentModel, self).__init__()
        
        # -----------------------
        # 1. 图像编码器 (ResNet-50)
        # -----------------------
        # 加载预训练的 ResNet50
        resnet = models.resnet50(pretrained=True)
        # 去掉最后的全连接层 (我们只需要特征，不需要ResNet原本的分类)
        # ResNet50 倒数第二层输出维度是 2048
        modules = list(resnet.children())[:-1] 
        self.image_encoder = nn.Sequential(*modules)
        self.img_feat_dim = 2048

        # -----------------------
        # 2. 文本编码器 (BERT)
        # -----------------------
        # 加载预训练 BERT (这里用英文版，如果是纯中文数据建议换 bert-base-chinese)
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_feat_dim = 768  # BERT base 的输出维度

        # -----------------------
        # 3. 融合层 (Fusion Layer)
        # -----------------------
        # 将图片特征(2048) + 文本特征(768) = 2816
        self.fusion_dim = self.img_feat_dim + self.text_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(512, num_classes) # 输出 3 个类别的概率 logits
        )

        # 选项：冻结预训练参数，只训练分类层 (如果显存太小爆了，可以把这里解开)
        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, images):
        """
        Args:
            input_ids: BERT的输入索引
            attention_mask: BERT的掩码
            images: 图片张量 (Batch, 3, 224, 224)
        """
        # --- 1. 提取图像特征 ---
        # images: [batch, 3, 224, 224] -> [batch, 2048, 1, 1]
        img_features = self.image_encoder(images)
        # 展平: [batch, 2048]
        img_features = img_features.view(img_features.size(0), -1)

        # --- 2. 提取文本特征 ---
        # BERT 输出: last_hidden_state, pooler_output
        # 我们取 pooler_output (对应 [CLS] token 的向量)，代表整句话的语义
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output # [batch, 768]

        # --- 3. 特征融合 ---
        # 拼接: [batch, 2048 + 768]
        fused_features = torch.cat((img_features, text_features), dim=1)

        # --- 4. 分类 ---
        logits = self.classifier(fused_features)
        
        return logits

# --- 简单的测试代码 ---
if __name__ == "__main__":
    print("正在初始化模型 (第一次运行会下载 BERT 和 ResNet 权重，请耐心等待)...")
    
    # 实例化模型
    model = MultimodalSentimentModel(num_classes=3)
    print("模型构建成功！")
    
    # 构造假数据测试一下维度是否匹配
    batch_size = 2
    # 假文本输入 (Batch=2, SeqLen=10)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 10))
    dummy_mask = torch.ones((batch_size, 10))
    # 假图片输入 (Batch=2, Channel=3, H=224, W=224)
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    outputs = model(dummy_input_ids.long(), dummy_mask.long(), dummy_images)
    
    print(f"输入尺寸: 文本{dummy_input_ids.shape}, 图片{dummy_images.shape}")
    print(f"输出尺寸: {outputs.shape}") # 应该是 [2, 3]
    
    if outputs.shape == (2, 3):
        print("模型测试通过！维度正确。")
    else:
        print("模型输出维度不对，请检查代码。")