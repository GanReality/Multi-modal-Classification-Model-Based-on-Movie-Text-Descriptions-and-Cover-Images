from model.Multimodel import *
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

start_time = time.time()
data_source = '../dataset_folder/dataset_Origin/test'
tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-base-chinese')
feature_extractor = ViTFeatureExtractor.from_pretrained('../pretrained/google-vit-base-patch16-224')
test_dataset = TextImageFolderDataset(data_source, tokenizer, feature_extractor)
dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)


weight = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalClassifier(768, 768, len(test_dataset.classes), weight, 'test').to(device)


model.eval()
all_labels = []
all_predicted_indices = []

all_count = [0,0]
right_count = [0,0]

with torch.no_grad():
    for i, sample in enumerate(dataloader):
        # 将输入数据移到设备上
        text_inputs = {k: v.squeeze(1).to(device) for k, v in sample['text_inputs'].items()}
        image_inputs = {k: v.squeeze(1).to(device) for k, v in sample['image_inputs'].items()}
        labels = sample['label'].to('cpu')

        outputs = model(text_inputs, image_inputs)

        probs = F.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(probs, dim=1)

        predicted_indices = predicted_indices.cpu().numpy()
        labels = labels.numpy()

        # 收集所有的真实标签和预测标签
        all_labels.extend(labels)
        all_predicted_indices.extend(predicted_indices)

        # 更新每个类别的正确预测数和总数
        for label, predicted in zip(labels, predicted_indices):
            if label == predicted:
                right_count[label] += 1
            all_count[label] += 1

# 计算 Accuracy, Precision, Recall, F1
accuracy = accuracy_score(all_labels, all_predicted_indices)
precision = precision_score(all_labels, all_predicted_indices, average='weighted')
recall = recall_score(all_labels, all_predicted_indices, average='weighted')
f1 = f1_score(all_labels, all_predicted_indices, average='weighted')

with open('multi.txt','a',encoding='utf-8') as file:
    file.write('acc:' + str(accuracy) + 'pre:' + str(precision) + 'rec:' + str(recall) + 'f1:' + str(f1) + '\n')

