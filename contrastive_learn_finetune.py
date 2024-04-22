from torch.utils.data import DataLoader, random_split
import torchvision
import torch
from utils import ScaleTrimap, test_finetuning_network, train_finetuning_network
from model import UNet
import argparse
import os
import numpy as np
import random


def load_model_with_filtered_state_dict(model, checkpoint_path, exclude_layers):
    # Load the full state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Get the current state dict from the model
    model_state_dict = model.state_dict()

    # Filter out layers from the loaded checkpoint state dict
    filtered_checkpoint = {k: v for k, v in checkpoint.items(
    ) if k in model_state_dict and model_state_dict[k].size() == v.size()}

    # Optionally, specifically exclude certain layers
    for layer in exclude_layers:
        if layer in filtered_checkpoint:
            del filtered_checkpoint[layer]

    # Load the filtered checkpoint into the model
    model.load_state_dict(filtered_checkpoint, strict=False)

    return model


seed_value = 16
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
availability = torch.cuda.is_available()
if availability:
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument("-pt", "--pretrain",
                    action=argparse.BooleanOptionalAction, default=False, help="Use pretrained model")
parser.add_argument("-p", "--patience", type=int, default=10,
                    help="Patience for early stopping")
parser.add_argument("-s", "--dataset-size", type=float, default=1.0, choices=[n/10 for n in range(1, 11)],
                    help="Fine-tuning dataset size ratio")
parser.add_argument("-b", "--batch-size", type=int, default=4,
                    help="Batch size for training")
parser.add_argument("-lc", "--load-checkpoint", type=int,
                    default=None, help="Load checkpoint")

args = parser.parse_args()



PRETRAIN = args.pretrain
SAMPLE_SIZE = args.dataset_size
SAMPLE_SIZE = 0.05
PRETRAIN_CHECKPOINT = args.load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
])
target_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
    ScaleTrimap(),
])

trainval_data = torchvision.datasets.OxfordIIITPet(
    download=True, root="./pet_dataset", transform=transform, target_transform=target_transform, target_types="segmentation")

pin_memory = torch.cuda.is_available()
pin_memory_device = device.type if pin_memory else ""
if SAMPLE_SIZE != 1.0:
    total_size = len(trainval_data)
    subset_size = int(total_size * SAMPLE_SIZE)
    discard_size = total_size - subset_size

    generator1 = torch.Generator().manual_seed(16)
    trainval_data, _ = random_split(
        trainval_data, [subset_size, discard_size], generator1)
    generator1 = torch.Generator().manual_seed(16)
    training_data, validation_data = random_split(
        trainval_data, [0.9, 0.1], generator1)
else:
    generator1 = torch.Generator().manual_seed(16)
    training_data, validation_data = random_split(
        trainval_data, [0.9, 0.1], generator1)

trainloader = DataLoader(training_data, batch_size=4,
                         shuffle=True, pin_memory=pin_memory, num_workers=4, pin_memory_device=pin_memory_device)

valloader = DataLoader(validation_data, batch_size=4,
                       shuffle=True, pin_memory=pin_memory, num_workers=4, pin_memory_device=pin_memory_device)

testing_data = torchvision.datasets.OxfordIIITPet(
    download=True, root="./pet_dataset", transform=transform, target_transform=target_transform, target_types="segmentation", split="test")
testloader = DataLoader(testing_data, batch_size=4,
                        shuffle=True, pin_memory=True, num_workers=4, pin_memory_device=device.type)


model = UNet(3, 3)
initial_weights = model.inc.double_conv[0].weight
print("Pretraining:  ", PRETRAIN)
if PRETRAIN:
    pretrain_dir = f"saved_models/contrastive_learning/pretrain/"
    last_model_path = ""
    for (dirpath, dirnames, filenames) in os.walk(pretrain_dir):
        last_model_path = os.path.join(
            pretrain_dir, f"{PRETRAIN_CHECKPOINT}_epochs.pt")
    exclude_layers = ['outc.conv.weight', 'outc.conv.bias']
    load_model_with_filtered_state_dict(model, last_model_path, exclude_layers)
    model.reinit_up()
    # model.freeze_down()

    print("Using pretrained model from:  ", last_model_path)

    MODEL_DIR = f"saved_models/contrastive_learning/finetune/{PRETRAIN_CHECKPOINT}_checkpoint/{str(SAMPLE_SIZE)}_size"
else:
    MODEL_DIR = f"saved_models/contrastive_learning/finetune/baseline/{str(SAMPLE_SIZE)}_size"

os.makedirs(MODEL_DIR, exist_ok=True)
model = model.to(device)

optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss().to(device)

MAX_EPOCHS = 200
# Pet, Background, Border
classes = (0, 1, 2)

print()
print("         Train           |  Val   |                                Test                                  ")
print(" Epoch |  Loss  |  Time  |  Loss  |  Loss  | IoU BG | IoU PET | IoU BD | IoU Mean |   Acc   | Dice Coeff ")
print("-------------------------|--------|----------------------------------------------------------------------")


patience = args.patience
no_improvement = 0
best_loss = torch.inf

for i in range(MAX_EPOCHS):
    train_loss, train_time = train_finetuning_network(
        model, optimiser, criterion, trainloader, device)
    val_loss, val_time, val_iou, _, _ = test_finetuning_network(
        model, criterion, valloader, device)
    test_loss, test_time, test_iou, test_acc, test_dice_loss = test_finetuning_network(
        model, criterion, testloader, device)

    print("{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}".format(
        str(i+1).center(7),
        str(train_loss).center(8),
        str(train_time).center(8),
        str(val_loss).center(8),
        str(test_loss).center(8),
        str(test_iou[1]).center(8),
        str(test_iou[0]).center(9),
        str(test_iou[2]).center(8),
        str(round(np.mean(test_iou), 5)).center(10),
        str(test_acc[0]).center(9),
        str(round(np.mean(test_dice_loss), 5)).center(12),
    ))

    if (i+1) % 20 == 0:
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, f"{i+1}_epochs.pt"))

    if val_loss < best_loss:
        best_loss = val_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement == patience:
        print(
            f"Stopping training at {i+1} epochs -> no improvement in validation loss in {patience} epochs")
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, f"{i+1}_epochs.pt"))
        break
