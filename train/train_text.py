from model.TextModel import *
import matplotlib.pyplot as plt
import time
import os
from torch import optim
import copy
from sklearn.metrics import precision_score, recall_score, f1_score


begin_time = time.time()

weight = ''
save_name = ''


current_dir = os.getcwd() + '\..\weights\weights_text'
if (os.listdir(current_dir) != []):
    existing_folders = [f for f in os.listdir(current_dir) if os.path.isdir(current_dir + '\\' + f)]
    folder_numbers = []
    for folder in existing_folders:
        number = int(folder)
        folder_numbers.append(number)
    max_number = max(folder_numbers)
    next_number = max_number + 1
    new_folder_name = str(next_number)
    path = current_dir + '\\' + new_folder_name
else:
    path = current_dir + '\\' + '1'
os.makedirs(path, exist_ok=True)


tokenizer = BertTokenizer.from_pretrained('../pretrained/bert-base-chinese')
train_dataset = TextFolderDataset('../dataset_folder/dataset_Origin/train', tokenizer)
val_dataset = TextFolderDataset('../dataset_folder/dataset_Origin/val', tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(768, len(train_dataset.classes), weight, 'train').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_loss_list = []
val_loss_list = []
val_accuracy = []
train_info = []
time_info = []

train_time = 0
val_time = 0
best_val = 0
best_state = copy.deepcopy(model.state_dict())
best_epoch = 0

epochs = 30

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    train_start_time = time.time()
    flag = 1
    for i, sample in enumerate(train_dataloader):
        text_inputs = {k: v.squeeze(1).to(device) for k, v in sample['text_inputs'].items()}
        labels = sample['label'].to(device)

        optimizer.zero_grad()

        outputs = model(text_inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_end_time = time.time()
    print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(train_dataloader):.4f}')
    print(f'Epoch[{epoch + 1}]训练花费的时间为{train_end_time - train_start_time}')
    train_time = train_time + train_end_time - train_start_time
    train_loss_list.append(running_loss / len(train_dataloader))

    model.eval()
    val_start_time = time.time()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, sample in enumerate(val_dataloader):
            text_inputs = {k: v.squeeze(1).to(device) for k, v in sample['text_inputs'].items()}
            labels = sample['label'].to(device)

            outputs = model(text_inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if (best_val < correct / total):
            best_val = correct / total
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        val_end_time = time.time()
        val_accuracy.append(100 * correct / total)
        print(f'Validation Loss: {val_loss / len(val_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')
        print(f'Epoch[{epoch + 1}]验证花费的时间为{val_end_time - val_start_time}')
        val_time = val_time + val_end_time - val_start_time
        val_loss_list.append(val_loss / len(val_dataloader))

    info = f'train_loss:{running_loss / len(train_dataloader):.4f}   val_loss:{val_loss / len(val_dataloader):.4f}   val_accuracy:{100 * correct / total:.3f}%'
    train_info.append(info)
    tinfo = f'Epoch[{epoch + 1}]  train:{(train_end_time - train_start_time):.1f}s   val:{(val_end_time - val_start_time):.1f}s'
    time_info.append(tinfo)

x = range(1, epochs + 1)
plt.figure()
plt.plot(x, train_loss_list, label='train_loss')
plt.plot(x, val_loss_list, label='val_loss')
plt.ylim(bottom = 0, top = 0.8)
plt.legend()
plt.title('Two Lines in One Plot')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.grid(True)
plt.xticks(range(1, epochs + 1, 5))  # 每5个间隔显示一次数字
plt.savefig(path + '\\' + save_name + '.png', dpi=300)
plt.show()

print('train finished')
print(f'验证集准确率最高出现在第{best_epoch + 1}次')

torch.save(model.state_dict(), path + '\\' + save_name + '.pth')
torch.save(best_state, path + '\\' + save_name + '_best' + '.pth')
with open(path + '\\' + save_name + '.txt', 'w', encoding='utf-8') as file:
    for i in range(len(train_info)):
        file.write(time_info[i] + '\n')
        file.write(train_info[i] + '\n')
        file.write('\n')

end_time = time.time()
print(f'一共花费的时间为:{end_time - begin_time}s,训练花费的时间为:{train_time}s,验证花费的时间为{val_time}s')

