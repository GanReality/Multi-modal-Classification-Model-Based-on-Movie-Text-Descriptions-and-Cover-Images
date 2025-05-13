from model.TextModel import *
import time
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

start_time = time.time()
data_source = '../dataset_folder/dataset_Origin/test'
tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-base-chinese')
test_dataset = TextFolderDataset(data_source, tokenizer)
dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)


weight = ''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(768, len(test_dataset.classes), weight, 'test').to(device)

all_count = [0,0]
right_count = [0,0]
model.eval()
all_labels = []
all_predicted_indices = []

with torch.no_grad():
    for i, sample in enumerate(dataloader):
        text_inputs = {k: v.squeeze(1).to(device) for k, v in sample['text_inputs'].items()}
        labels = sample['label'].to('cpu')

        outputs = model(text_inputs)
        probs = F.softmax(outputs, dim=1)
        _, predicted_indices = torch.max(probs, dim=1)

        predicted_indices = predicted_indices.cpu().numpy()
        labels = labels.numpy()

        all_labels.extend(labels)
        all_predicted_indices.extend(predicted_indices)

        for label, predicted in zip(labels, predicted_indices):
            if label == predicted:
                right_count[label] += 1
            all_count[label] += 1

# 计算 Accuracy, Precision, Recall, F1
accuracy = accuracy_score(all_labels, all_predicted_indices)
precision = precision_score(all_labels, all_predicted_indices, average='weighted')
recall = recall_score(all_labels, all_predicted_indices, average='weighted')
f1 = f1_score(all_labels, all_predicted_indices, average='weighted')

with open('text.txt','a',encoding='utf-8') as file:
    file.write('acc:' + str(accuracy) + 'pre:' + str(precision) + 'rec:' + str(recall) + 'f1:' + str(f1) + '\n')
