from utils import ScaleTrimap, IoU, DiceCoefficient, PixelAccuracy
from torch.utils.data import DataLoader, random_split
import torchvision
import torch
from model import UNet
import numpy as np
import time
from utils import ScaleTrimap, test_finetuning_network, train_finetuning_network
import os
import argparse

def set_seed():
    seed_value=16
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def finetune(model, checkpoint):
    print(f"\n\n==================== Fine tuning on checkpoint = {checkpoint} ====================")
    MODEL_DIR = f"finetune_final_results/contrastive_learning/{checkpoint}_epochs"
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
    print("-------------------------|--------|--------------------------------------------------------------------- ")

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

        if (i+1) % 5 == 0:
            torch.save(model.state_dict(),
                    os.path.join(MODEL_DIR, f"{i+1}_epochs.pt"))

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement == patience:
            print(f"Stopping training at {i+1} epochs -> no improvement in validation loss in {patience} epochs")
            torch.save(model.state_dict(),
                    os.path.join(MODEL_DIR, f"{i+1}_epochs.pt"))
            break



def load_model_with_filtered_state_dict(model, checkpoint_path, exclude_layers):
    # Load the full state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Get the current state dict from the model
    model_state_dict = model.state_dict()

    # Filter out layers from the loaded checkpoint state dict
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}

    # Optionally, specifically exclude certain layers
    for layer in exclude_layers:
        if layer in filtered_checkpoint:
            del filtered_checkpoint[layer]

    # Load the filtered checkpoint into the model
    model.load_state_dict(filtered_checkpoint, strict=False)

    return model

parser = argparse.ArgumentParser()
parser.add_argument("-pt", "--pretrain",
                    action=argparse.BooleanOptionalAction, default=False)

parser.add_argument("-p", "--patience", type=int, default=5,
                    help="Patience for early stopping")

parser.add_argument("-c", "--list_of_integers", nargs='+', type=int,
                    help="List of integers")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

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
    download=True, root="./pet_dataset", transform=transform,  target_transform= target_transform, target_types="segmentation")

generator1 = torch.Generator().manual_seed(16)
training_data, validation_data = random_split(trainval_data, [0.9, 0.1], generator1)

trainloader = DataLoader(training_data, batch_size=64,
                         shuffle=True, pin_memory=True, num_workers=4, pin_memory_device=device.type)

valloader = DataLoader(validation_data, batch_size=64,
                       shuffle=True, pin_memory=True, num_workers=4, pin_memory_device=device.type)

testing_data = torchvision.datasets.OxfordIIITPet(
    download=True, root="./pet_dataset", transform=transform, target_transform=target_transform, target_types="segmentation", split="test")
testloader = DataLoader(testing_data, batch_size=64,
                        shuffle=True, pin_memory=True, num_workers=4, pin_memory_device=device.type)

pretrain_dir = f"saved_models/contrastive_learning_new/pretrain/"

for (_, _, filenames) in os.walk(pretrain_dir):
    checkpoints = sorted([int(s.split("_")[0])
                    for s in filenames if s.endswith(".pt")])
    for checkpoint in args.list_of_integers:
        model_path = os.path.join(pretrain_dir, f"{checkpoint}_epochs.pt")
        set_seed()
        model = UNet(3, 3)
        # model.load_state_dict(torch.load(model_path))
        exclude_layers = ['outc.conv.weight', 'outc.conv.bias']
        load_model_with_filtered_state_dict(model, model_path, exclude_layers)
        model.reinit_up()
        print("Using pretrained model from:  ", model_path)
        finetune(model, checkpoint)


