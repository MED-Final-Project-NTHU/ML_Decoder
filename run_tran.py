import torch
from transformer_model import transformer_model
import numpy as np
import os
import random
import torch.optim as optim
import torch.nn as nn
from Myloader import *
import time
import torchvision.models as models
from torchmetrics.classification import MultilabelAveragePrecision
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(model, val_loader):
    model.eval()
    test_running_loss = 0.0
    test_total = 0

    with torch.no_grad():
        record_target_label = torch.zeros(1, 19).to(device)
        record_predict_label = torch.zeros(1, 19).to(device)
        for (test_imgs, test_labels, test_dicoms) in val_loader:
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            test_labels = test_labels.squeeze(-1)

            test_output = model(test_imgs)
            loss = criterion(test_output, test_labels)

            test_running_loss += loss.item() * test_imgs.size(0)
            test_total += test_imgs.size(0)

            record_target_label = torch.cat((record_target_label, test_labels), 0)
            record_predict_label = torch.cat((record_predict_label, test_output), 0)


        record_target_label = record_target_label[1::]
        record_predict_label = record_predict_label[1::]

        metric = MultilabelAveragePrecision(num_labels=19, average="macro", thresholds=None)
        mAP = metric(record_predict_label, record_target_label.to(torch.int32))

    return mAP, test_running_loss, test_total


class XRay_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")  # Assuming RGB images, adjust accordingly
        labels = torch.tensor(self.data_frame.iloc[idx, 1:].values.astype(float), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
    

img_path = 'data/images/'
csv_path = 'data/records2.csv'

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.RandomRotation(10),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])


rnd_rot = transforms.RandomRotation(10)
rnd_rs_crop = transforms.RandomResizedCrop(256, scale=(0.9, 1.0))

rnd_ps = transforms.RandomPosterize(bits=6, p=0.2)
rnd_c = transforms.RandomCrop(size=(256, 256), padding=(10, 10))
rnd_hsv = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)
blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.001, 0.6))
rnd_b = transforms.RandomApply([blurrer], p=0.2)

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        rnd_rot,
        rnd_c,
        rnd_ps,
        rnd_rs_crop,
        rnd_hsv,
        rnd_b,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

dataset = XRay_Dataset(csv_file=csv_path, transform=transform)


set_seed(123)
#     weight_dir = ""
#     if not os.path.exists(weight_dir):
#         os.makedirs(weight_dir)

epochs = 10
batch_size = 32
num_classes = 19

weight_path = "weights_bal/"

train_path = "data/MICCAI_long_tail_train.tfrecords"
train_index = "data/MICCAI_long_tail_train.tfindex"
val_path = "data/MICCAI_long_tail_val.tfrecords"

val_index = "data/MICCAI_long_tail_val.tfindex"
opt_lr = 3e-5
weight_decay = 1e-5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
encoder = transformer_model().to(device)
# checkpoint = torch.load(weight_path + "/2epoch_163840step.pt")
# encoder.load_state_dict(checkpoint['model_state_dict'])
# weights = [1.0, 3.0, 1.0, 2.0, 1.0, 2.0, 2.0, 5.0, 1.0, 1.0, 1.0, 5.0, 3.0, 2.0, 5.0, 1.0, 1.0, 1.0, 4.0]
# sampler = WeightedRandomSampler(weights, len(weights))

opt = optim.Adam(encoder.parameters(), lr=opt_lr, weight_decay = weight_decay)
# opt.load_state_dict(checkpoint['optimizer_state_dict'])
# train_loader = Myloader(train_path, train_index, batch_size, num_workers=0, shuffle=True)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = Myloader(val_path, val_index, batch_size, num_workers=0, shuffle=False)

criterion = nn.BCEWithLogitsLoss()


train_losses = []
test_losses = []


max_map = 0
total = 0
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    encoder.train()
    running_loss = 0.0
    start_time = time.time()
    count = 0

    for (imgs, labels) in train_loader:
        encoder.zero_grad()
        opt.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)
        labels = labels.squeeze(-1)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = encoder(imgs)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        count += imgs.size(0)

        if count != 0 and count % 1024 == 0 and total == 0:
            print(f"epoch {epoch}: {count}/unknown finished / train loss: {running_loss / count}")

        elif count != 0 and count % 10 == 0 and total != 0:
            print(f"epoch {epoch}: {count}/{total} (%.2f %%) finished / train loss: {running_loss / count}" % (count/total))
        
        if count % 32768 == 0 and count != 0:
            torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, weight_path+"/{}epoch_{}step.pt".format(epoch, count))

    total = count
    mAP, test_running_loss, test_total = evaluate(encoder, val_loader)
    
    train_losses.append(running_loss / count)
    test_losses.append(test_running_loss)
    
    if mAP > max_map:
        max_map = mAP
        torch.save({
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, f"{weight_path}/model_best.pt")
    if epoch % 10 == 0:
        torch.save({
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, weight_path+"/{}epoch.pt".format(epoch))

    end_time = time.time()
    duration = end_time - start_time

    print(f"epoch {epoch} / mAP: {mAP} / test loss: {test_running_loss / test_total} / duration: {duration}")