from model import UNetContrastive
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torch
import numpy as np
from utils import CocoImages
import time
from utils import LARS, SimCLR_Loss
import argparse


def set_seed(seed=16):
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch-size", type=int, default=4,
                    help="Batch size for training")
parser.add_argument("-lc", "--load-checkpoint", type=int,
                    default=None, help="Load checkpoint")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CocoDataGen(Dataset):
    def __init__(self, phase, dataset, s=0.5):
        self.phase = phase
        self.dataset = dataset
        self.s = s
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(32, (0.8, 1.0)),
            transforms.Compose([
                transforms.RandomApply([
                    transforms.ColorJitter(
                        0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]  # Load image

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32) / 255.0)

        x1 = self.transforms(x)
        x2 = self.transforms(x)

        return x1, x2


standard_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
])


COCO_ROOT = "./coco_dataset/raw"
train_dataset = CocoImages(
    image_root=COCO_ROOT, transform=standard_transform, split="train")
train_dg = CocoDataGen(phase='train', dataset=train_dataset)

val_dataset = CocoImages(image_root=COCO_ROOT,
                         transform=standard_transform, split="val")
val_dg = CocoDataGen(phase='val', dataset=val_dataset)

# Create data loaders
train_loader = DataLoader(train_dg, batch_size=128, shuffle=True,
                          num_workers=4, pin_memory=True, pin_memory_device=device.type)
val_loader = DataLoader(val_dg, batch_size=128, shuffle=True,
                        num_workers=4, pin_memory=True, pin_memory_device=device.type)


model = UNetContrastive(n_channels=3, embedding_dim=256)
model.to(device)
optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)

# "decay the learning rate with the cosine decay schedule without restarts"
# SCHEDULER OR LINEAR EWARMUP
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: (epoch+1)/10.0, verbose=True)

# SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 500, eta_min=0.05, last_epoch=-1, verbose=True)

# LOSS FUNCTION
criterion = SimCLR_Loss(batch_size=128, temperature=0.5)


NUM_EPOCHS = 100
best_loss = float('inf')


nr = 0
tr_loss = []
val_loss = []
set_seed(16)
for epoch in range(NUM_EPOCHS):
    f = open("saved_models/contrastive_learning/training_log.txt", "a")
    f.write(f"\n\nEpoch [{epoch}/{NUM_EPOCHS}]\t")
    f.close()
    stime = time.time()
    tr_loss_epoch = 0
    model.train()
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device=device).float()
        x_j = x_j.to(device=device).float()

        # positive pair with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)  # Adjusted criterion usage
        loss.backward()

        optimizer.step()

        if nr == 0 and step % 50 == 0:
            print(
                f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

        tr_loss_epoch += loss.item()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    lr = optimizer.param_groups[0]["lr"]

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss_epoch = 0
        for step, (x_i, x_j) in enumerate(val_loader):
            x_i = x_i.to(device=device).float()
            x_j = x_j.to(device=device).float()

            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)

            if nr == 0 and step % 50 == 0:
                print(
                    f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

            val_loss_epoch += loss.item()

    time_taken = (time.time()-stime)/60
    if nr == 0:
        f = open("saved_models/contrastive_learning/training_log.txt", "a")
        tr_loss.append(tr_loss_epoch / len(train_loader))
        val_loss.append(val_loss_epoch / len(val_loader))
        f.write(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
        f.write(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Validation Loss: {val_loss_epoch / len(val_loader)}\t lr: {round(lr, 5)}")
        f.write(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Time Taken: {time_taken} minutes")
        f.close()
        torch.save(model.state_dict(
        ), f"saved_models/contrastive_learning/contrastive-learning-{epoch+1}_EPOCHS.pt")

    # Save best model if validation loss improves
    avg_val_loss = val_loss_epoch / len(val_loader)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(
        ), f"saved_models/contrastive_learning/lowest_val_model-{epoch+1}_EPOCHS.pt")
        print("Saved new best model")
