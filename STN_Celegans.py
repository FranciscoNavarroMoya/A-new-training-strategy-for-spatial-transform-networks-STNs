import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()
print(device)

path = '/home/ingivision/PycharmProjects/STN/La_Fe2/Micro2_trans_2'
images = os.listdir(path)
ori_images = []
target_images = []

for i in range(len(images)):
  ori_images.append(f"Dist_Micro2/{images[i]}")
  target_images.append(f"Dist_Micro1/{images[i]}")


x_train = ori_images
y_train = target_images
random.seed(2020)
batch_size = 8
epochs = 500
path = '/home/ingivision/PycharmProjects/STN/github/data'

class transf(Dataset):
    def __init__(self, images, labels, transform=None):

        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        dist = Image.open(f"{path}/{self.images[item]}")
        target_dist = Image.open(f"{path}/{self.labels[item]}")
        dist = np.asarray(dist)
        target_dist = np.asarray(target_dist)
        x,y = self.images[item].split("/")
        seg = Image.open(f"{path}/Seg_Micro2/{y}")
        target_seg = Image.open(f"{path}/Seg_Micro1/{y}")
        seg = np.asarray(seg)
        target_seg = np.asarray(target_seg)
        if self.transform:
            dist = self.transform(dist)
            seg = self.transform(seg)
            target_seg = self.transform(target_seg)
            target_dist = self.transform(target_dist)

        return dist, target_dist, seg, target_seg

transform=transforms.Compose([
                                  transforms.ToTensor(),
                 ])

train_ds = transf(x_train,
                y_train,
                transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

class AffineNet(nn.Module):
    def __init__(self, input_nc=1, transform=torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)):
        super(AffineNet, self).__init__()
        # Spatial transformer localization-network

        self.affine_stl = nn.Sequential(
            nn.Conv2d(input_nc, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.SELU(True),
            nn.Conv2d(40, 10, kernel_size=3),
            nn.SELU(True),
        )

        self.stl_fc = self.fc_loc = nn.Sequential(
            nn.Linear(182120, 32),
            nn.SELU(True),
            nn.Linear(32, 4 * 3)
        )

        self.stl_fc[2].weight.data.zero_()
        self.stl_fc[2].bias.data.copy_(transform)

    def forward(self, x):
        xs = self.affine_stl(x)
        xs = xs.view(-1, 182120)
        theta = self.stl_fc(xs)
        self.theta = theta.view(-1, 3, 4)
        grid = F.affine_grid(self.theta, (x.size(0),1,1,972, 1296))

        return grid, theta.cpu()


model = AffineNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_fun = nn.MSELoss(reduction='mean')
criterion = loss_fun.to(device)


def train_dist(model, dataloader, epoch):
    model.train()
    running_loss = []
    iou = 0
    for step, (dist, target_dist, seg, target_seg) in enumerate(tqdm(dataloader, desc="Training")):
        dist = dist.to(device)
        seg = seg.to(device)
        target_dist = target_dist.to(device)
        target_seg = target_seg.to(device)

        optimizer.zero_grad()
        grid, theta = model(dist)
        out = F.grid_sample(dist.view(-1, 1, 1, 1944, 2592), grid, padding_mode="border")
        out = out.view(-1, 1, 972, 1296)

        loss = criterion(out, target_dist)
        loss.backward()
        optimizer.step()

        out2 = F.grid_sample(seg.view(-1, 1, 1, 1944, 2592), grid, padding_mode="border")
        out2 = out2.view(-1, 1, 972, 1296)

        loss = criterion(out2, target_seg)
        running_loss.append(loss.cpu().detach().numpy())

        for i in range(len(out2)):
            t, thresh = cv2.threshold(out2[i][0].cpu().detach().numpy()*255, 127, 255, cv2.THRESH_BINARY)
            img1 = 255-thresh.astype(np.uint8)
            ref = 255-target_seg[i][0].cpu().detach().numpy()*255
            intersection = np.logical_and(ref, img1)
            union = np.logical_or(ref, img1)
            iou += np.sum(intersection) / np.sum(union)

    return (np.mean(running_loss),iou/len(x_train))

def train_seg(model, dataloader, epoch):
    model.train()
    running_loss = []
    iou = 0
    for step, (dist, target_dist, seg, target_seg) in enumerate(tqdm(dataloader, desc="Training")):
        seg = seg.to(device)
        target_seg = target_seg.to(device)

        optimizer2.zero_grad()

        grid2, theta = model(seg)
        out2 = F.grid_sample(seg.view(-1, 1, 1, 1944, 2592), grid2, padding_mode="border")
        out2 = out2.view(-1, 1, 972, 1296)

        loss = criterion(out2, target_seg)
        if loss > 0.05:
            continue
        loss.backward()
        optimizer2.step()
        running_loss.append(loss.cpu().detach().numpy())

        for i in range(len(out2)):
            t, thresh = cv2.threshold(out2[i][0].cpu().detach().numpy()*255, 127, 255, cv2.THRESH_BINARY)
            img1 = 255-thresh.astype(np.uint8)
            ref = 255-target_seg[i][0].cpu().detach().numpy()*255
            intersection = np.logical_and(ref, img1)
            union = np.logical_or(ref, img1)
            iou += np.sum(intersection) / np.sum(union)

    return(np.mean(running_loss), iou/len(x_train))


l = 10000000
m = 10000000
n = 10000000
o = 10000000
iou = 0

train_loss = []
val_loss = []
seg_not_improve = 0
epoch_change = 0

for epoch in tqdm(range(epochs), desc="Epochs"):
    print(f"Epoch {epoch + 1} of {epochs}")

    if iou < 0.6:
        # train_epoch_loss, iou = train_dist(model, train_loader, epoch)
        if epoch%2 == 0:
            train_epoch_loss, iou = train_dist(model, train_loader, epoch)
            print(iou)
        else:
            if epoch == 1:
                optimizer2 = optim.Adam(model.parameters(), lr=0.0001)
            train_epoch_loss, iou = train_seg(model, train_loader, epoch)
            print(iou)
        train_loss.append(train_epoch_loss)
    else:
        if epoch_change == 0:
            epoch_change = epoch + 1
        if epoch == 1:
            optimizer2 = optim.Adam(model.parameters(), lr=0.0001)

        train_epoch_loss, iou = train_seg(model, train_loader, epoch)
        print(iou)
        train_loss.append(train_epoch_loss)
        if train_epoch_loss < m:
            m = train_epoch_loss
            seg_not_improve = 0
            torch.save(model.state_dict(), '/home/ingivision/PycharmProjects/STN/Ensayos para documentacion/Modelos/Gusanos/best_modelSTN_train_distseg.pth')
        else:
            seg_not_improve += 1
            if seg_not_improve == 20:
                epoch_fin = epoch - 20
        if (seg_not_improve >= 20) or iou > 0.86:
            break

    print("Train_Loss: {:.6f}".format(train_epoch_loss))

total = 0
mine = 10000000000000
maxe = 0
iou = 0
miniou = 1
maxiou = 0
model.load_state_dict(torch.load('/home/ingivision/PycharmProjects/STN/Ensayos para documentacion/Modelos/Gusanos/best_modelSTN_train_distseg.pth'))
test_results = []
loss = []
model.eval()
num = len(x_train)
with torch.no_grad():
    for (dist, target_dist, seg, target_seg) in tqdm(train_loader):
        grid, theta= model(seg.to(device))
        grid = grid.cpu()
        results = F.grid_sample(seg.view(-1,1,1,1944,2592), grid, padding_mode="border")
        loss_i = loss_fun(results.view(-1, 1, 972, 1296), target_seg)
        loss.append(loss_i.cpu().detach().numpy())
        #
        for i in range(len(dist)):
            a = 0
            ref = 255-target_seg[i][0].detach().numpy()*255
            img1 = results[i][0][0].detach().numpy()*255
            ref = ref.astype(np.uint8)
            img1 = img1.astype(np.uint8)
            t, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
            thresh1 = 255-thresh1
            intersection = np.logical_and(ref, thresh1)
            union = np.logical_or(ref, thresh1)
            iou_a = np.sum(intersection) / np.sum(union)
            img = np.asarray(ref, dtype=np.int32)
            thresh1 = np.asarray(thresh1, dtype=np.int32)

            diff1 = np.abs(ref - thresh1)

            a = int(sum(sum(diff1)) / 255)

            if a<mine:
                mine = a
            if a>maxe:
                maxe = a

            iou += iou_a
            if iou_a<miniou:
                miniou = iou_a
            if iou_a>maxiou:
                maxiou = iou_a

            total += a


media = total/num

iou = iou/num

print(num)
loss = np.mean(loss)
print(loss)
print("Train loss: {:.6f}".format(np.mean(loss)))
print(total, media, mine, maxe, iou, miniou, maxiou)

print("Epoca de cambio de modelo: {:.1f}".format(epoch_change))

plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/ingivision/PycharmProjects/STN/Ensayos para documentacion/gráficas de loss/entrenamiento intercalado/new_loss.png')
plt.show()