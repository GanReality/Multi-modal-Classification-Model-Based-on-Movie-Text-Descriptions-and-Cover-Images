import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
import os
import glob

class TextClassifier(nn.Module):
    def __init__(self, text_dim, num_classes, weight, type):
        super(TextClassifier, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-chinese')
        self.fc = nn.Linear(text_dim, num_classes)

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
            self.text_model.eval()
            print('eval')
        elif type == 'train':
            self.text_model.train()
            print('train')

        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, text_inputs):
        text_outputs = self.text_model(**text_inputs)
        x = self.fc(text_outputs.last_hidden_state[:, 0, :])
        return x


class TextFolderDataset(Dataset):
    def __init__(self, root_dir, tokenizer):
        self.root_dir = root_dir
        self.tokenizer = tokenizer


        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        print(self.class_to_idx)

        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for txt_path in glob.glob(os.path.join(class_dir, '*.txt')):
                if os.path.exists(txt_path):
                    self.samples.append((txt_path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        txt_path, label = self.samples[idx]

        with open(txt_path, 'r', encoding='UTF-8', errors='ignore') as f:
            text = f.read().strip()
        text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)

        return {'text_inputs': text_inputs, 'label': label}
