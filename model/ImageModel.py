import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import os
from PIL import Image
import glob
import chardet

class ImageClassifier(nn.Module):
    def __init__(self, image_dim, num_classes, weight, type):
        super(ImageClassifier, self).__init__()
        self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.image_norm = nn.LayerNorm(image_dim)
        # self.fc1 = nn.Linear(image_dim, 256)
        # self.fc2 = nn.Linear(256, 32)
        # self.fc3 = nn.Linear(32, num_classes)
        self.fc = nn.Linear(image_dim, num_classes)
        # 尝试加载整个模型的预训练权重
        if (weight != ''):
            if os.path.exists(weight):
                pretrained_dict = torch.load(weight)
                # 获取整个模型的state_dict
                model_dict = self.state_dict()
                # 选择与model_dict键名匹配的部分
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                if len(pretrained_dict):
                    print('OK')
                # 更新整个模型的state_dict
                model_dict.update(pretrained_dict)
                # 加载整个模型的预训练权重
                self.load_state_dict(model_dict)
                print('Entire model weights already loaded.')

        if type == 'test':
            self.image_model.eval()
            print('eval')
        elif type == 'train':
            self.image_model.train()
            print('train')

        for param in self.image_model.parameters():
            param.requires_grad = False

    def forward(self, image_inputs):
        image_outputs = self.image_model(**image_inputs)
        x = self.fc(image_outputs.last_hidden_state[:, 0, :])
        # x = self.fc1(image_outputs.pooler_output)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        # 获取所有子目录，每个子目录对应一个类别
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # self.class_to_idx = {'normal': 0, 'porn': 1}
        # print(self.class_to_idx)
        # 收集所有图像和文本文件及其对应的标签
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):  # 假设文本和图像文件名相同，扩展名不同
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 加载和预处理文本
        # detected_encoding = detect_encoding(txt_path)
        # print("Detected Encoding:", detected_encoding)

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # 返回文本、图像特征和标签
        return {'image_inputs': image_inputs, 'label': label}
