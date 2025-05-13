import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import os
from PIL import Image
import glob
import chardet

class MultimodalClassifier(nn.Module):
    def __init__(self, text_dim, image_dim, num_classes, weight, type):
        super(MultimodalClassifier, self).__init__()
        self.text_model = BertModel.from_pretrained('../pretrained/bert-base-chinese')
        self.image_model = ViTModel.from_pretrained('../pretrained/google-vit-base-patch16-224')
        self.text_norm = nn.LayerNorm(text_dim)
        self.image_norm = nn.LayerNorm(image_dim)
        # self.fc1 = nn.Linear(text_dim + image_dim, 512)
        # self.fc2 = nn.Linear(512, 32)
        # self.fc3 = nn.Linear(32, 2)
        self.fc1 = nn.Linear(text_dim + image_dim, num_classes)

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
            else:
                print('weight is not existing')

        if type == 'test':
            self.text_model.eval()
            self.image_model.eval()
            print('eval')
        elif type == 'train':
            self.text_model.train()
            self.image_model.train()
            print('train')

        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.image_model.parameters():
            param.requires_grad = False

    def forward(self, text_inputs, image_inputs):
        text_outputs = self.text_model(**text_inputs)
        image_outputs = self.image_model(**image_inputs)
        text_features = text_outputs.last_hidden_state[:, 0, :] # 取[CLS]标记的输出作为文本特征
        image_features = image_outputs.pooler_output  # 使用ViT的pooler_output作为图像特征
        # image_features = image_outputs.last_hidden_state[:, 0, :]
        # text_features = self.text_norm(text_features)
        # image_features = self.image_norm(image_features)
        x = torch.cat((text_features, image_features), dim=1)
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


class TextImageFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer, feature_extractor):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
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
            for txt_path in glob.glob(os.path.join(class_dir, '*.txt')):
                img_path = txt_path.replace('.txt', '.jpg')  # 假设文本和图像文件名相同，扩展名不同
                if os.path.exists(img_path):
                    self.samples.append((txt_path, img_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_path, img_path, label = self.samples[idx]
        # 加载和预处理文本
        # detected_encoding = detect_encoding(txt_path)
        # print("Detected Encoding:", detected_encoding)

        with open(txt_path, 'r', encoding='UTF-8', errors='ignore') as f:
            text = f.read().strip()
        text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)



        # 加载和预处理图像
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_inputs = self.feature_extractor(images=image, return_tensors="pt")

        # 返回文本、图像特征和标签
        return {'text_inputs': text_inputs, 'image_inputs': image_inputs, 'label': label}



# class Evaluate_Dataset(Dataset):
#     def __init__(self, root_dir, tokenizer, feature_extractor, page):
#         self.root_dir = root_dir + '\\' + f'{page}'
#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor
#         self.samples = []
#
#         for txt_path in glob.glob(os.path.join(self.root_dir, '*.txt')):
#             img_path = txt_path.replace('.txt', '.jpg')
#             if os.path.exists(img_path):
#                 self.samples.append((txt_path, img_path))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         try:
#             txt_path, img_path = self.samples[idx]
#             # 加载和预处理文本
#             # detected_encoding = detect_encoding(txt_path)
#             # print("Detected Encoding:", detected_encoding)
#
#             with open(txt_path, 'r', encoding='UTF-8', errors='ignore') as f:
#                 text = f.read().strip()
#             text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
#
#             # 加载和预处理图像
#             image = Image.open(img_path)
#             if image.mode != 'RGB':
#                 image = image.convert('RGB')
#             image_inputs = self.feature_extractor(images=image, return_tensors="pt")
#
#             # 返回文本、图像特征和标签
#             return {'text_inputs': text_inputs, 'image_inputs': image_inputs, 'txt_path': txt_path[0:len(txt_path)]}
#         except Exception as e:
#             print(e)


class Evaluate_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, tokenizer, feature_extractor, page):
        self.root_dir = os.path.join(root_dir, f'{page}')
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.samples = []

        for txt_path in glob.glob(os.path.join(self.root_dir, '*.txt')):
            img_path = txt_path.replace('.txt', '.jpg')
            if os.path.exists(img_path):
                self.samples.append((txt_path, img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_path, img_path = self.samples[idx]

        # 尝试加载和预处理文本
        try:
            with open(txt_path, 'r', encoding='UTF-8', errors='ignore') as f:
                text = f.read().strip()
            text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
        except Exception as e:
            print(f"Failed to load text from {txt_path}: {e}")
            return {'text_inputs': {}, 'image_inputs': {}, 'txt_path': txt_path}  # 跳过这个样本

        # 尝试加载和预处理图像
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_inputs = self.feature_extractor(images=image, return_tensors="pt")
        except Exception as e:
            print(f"Failed to load image from {img_path}: {e}")
            return {'text_inputs': {}, 'image_inputs': {}, 'txt_path': txt_path}  # 跳过这个样本

        # 返回文本、图像特征和标签
        return {'text_inputs': text_inputs, 'image_inputs': image_inputs, 'txt_path': txt_path}


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding']

