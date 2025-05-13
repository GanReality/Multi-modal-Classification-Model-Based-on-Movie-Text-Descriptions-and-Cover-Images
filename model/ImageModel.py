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

        self.fc = nn.Linear(image_dim, num_classes)

        if (weight != ''):
            if os.path.exists(weight):
                pretrained_dict = torch.load(weight)

                model_dict = self.state_dict()

                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                if len(pretrained_dict):
                    print('OK')

                model_dict.update(pretrained_dict)

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
        return x


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        # 获取所有子目录，每个子目录对应一个类别
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_inputs = self.feature_extractor(images=image, return_tensors="pt")

        return {'image_inputs': image_inputs, 'label': label}
