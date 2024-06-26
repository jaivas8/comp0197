import torch
from typing import Any
from torch.utils.data import DataLoader
import time
from .metrics import IoU, mse, PixelAccuracy, DiceCoefficient

torch.manual_seed(42)

def train_finetuning_network(net: torch.nn.Module, optimiser: torch.optim.Optimizer, criterion: Any, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    running_loss = torch.zeros(1).to(device)
    epoch_start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad(set_to_none=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets.squeeze(dim=1))
        loss.backward()
        optimiser.step()
        running_loss += loss
    epoch_end_time = time.time()
    return round((running_loss/(batch_idx+1)).item(), 5), round(epoch_end_time-epoch_start_time, 0)


def test_finetuning_network(net: torch.nn.Module, criterion: Any, dataloader: DataLoader, device: torch.device) -> tuple[float]:
    with torch.no_grad():
        running_loss = torch.zeros(1).to(device)
        running_iou = torch.zeros(3).to(device)
        running_accuracy = torch.zeros(1).to(device)
        running_dice_loss = torch.zeros(3).to(device)
        epoch_start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.squeeze(dim=1))
            running_loss += loss
            predictions = torch.argmax(outputs, dim=1)
            running_iou += IoU(predictions, targets)
            running_accuracy += PixelAccuracy(predictions, targets)
            running_dice_loss += DiceCoefficient(predictions, targets)
        epoch_end_time = time.time()
    return round((running_loss/(batch_idx+1)).item(), 5), round(epoch_end_time-epoch_start_time, 0), [round(iou.item(), 5) for iou in running_iou/batch_idx], [round(acc.item(), 5) for acc in running_accuracy/batch_idx], [round(dice_loss.item(), 5) for dice_loss in running_dice_loss/batch_idx]

def test_pretraining_network(net: torch.nn.Module, criterion: Any, dataloader: DataLoader, device: torch.device):
    with torch.no_grad():
        running_loss = torch.zeros(1).to(device)
        running_mse = torch.zeros(1).to(device)
        epoch_start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.squeeze(dim=1))
            running_loss += loss
            running_mse += mse(outputs, targets)
        epoch_end_time = time.time()
    return round((running_loss/batch_idx).item(), 5), round(epoch_end_time-epoch_start_time, 0), round(running_mse.item()/(batch_idx+1), 5)
