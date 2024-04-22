from utils import Mask, CocoImages
from torch.utils.data import DataLoader
import torchvision
import torch
from model import UNet
from utils import test_pretraining_network, train_finetuning_network
import argparse
import os

masking_enum = ["grid", "pixel", "random_erasing"]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mask-method", type=str, choices=masking_enum,
                    required=True, help="Masking method to use")
parser.add_argument("-mr", "--mask-ratio", type=float, required=True,
                    choices=[n/10 for n in range(1, 10)], help="Masking ratio")
parser.add_argument("-lc", "--load-checkpoint", type=int,
                    default=None, help="Load checkpoint")
parser.add_argument("-p", "--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument('-dir', "--save_directory", required=True, type=str)
args = parser.parse_args()


MASK_METHOD = args.mask_method
MASK_RATIO = args.mask_ratio

# MODEL_DIR = f"saved_models/pretrain/{MASK_METHOD}_{int(MASK_RATIO*100)}"
MODEL_DIR = os.path.join(args.save_directory, f"{MASK_METHOD}_{int(MASK_RATIO*100)}")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():   
    torch.backends.cudnn.benchmark = True


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
    Mask(MASK_RATIO),
])

COCO_ROOT = "./coco_dataset/raw"

pin_memory = torch.cuda.is_available()  
pin_memory_device = device.type if pin_memory else ""

trainset = CocoImages(
    image_root=COCO_ROOT, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True,
                         num_workers=4, pin_memory_device=pin_memory_device, pin_memory=pin_memory)

validationset = CocoImages(
    image_root=COCO_ROOT, transform=transform, split="val")
validationloader = DataLoader(
    validationset, batch_size=32, shuffle=True, num_workers=4, pin_memory_device=pin_memory_device, pin_memory=pin_memory)


if args.load_checkpoint:
    model = UNet(3, 3).to(device)
    checkpoint_dir = os.path.join(
        MODEL_DIR, f"{args.load_checkpoint}_epochs.pt")
    if not os.path.exists(checkpoint_dir):
        raise Exception("Specified Check Point does not exists")
    dir_to_load = checkpoint_dir
    model.load_state_dict(torch.load(dir_to_load))
    print(f"Loaded model from {dir_to_load}")
    start_epoch = args.load_checkpoint
else:
    model = UNet(3, 3).to(device)
   
    start_epoch = 0

optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.L1Loss()

MAX_EPOCHS = 50
print("Masking method: ", MASK_METHOD)
print("Masking ratio:  ", MASK_RATIO)

print()
print(" Epoch |  Tr Loss  |  Te Loss  |  Time  |  MSE  ")
print("------------------------------------------------")

patience = args.patience
no_improvement = 0
best_loss = torch.inf

for i in range(start_epoch, MAX_EPOCHS):
    train_loss, train_time = train_finetuning_network(
        model, optimiser, criterion, trainloader, device)
    test_loss, test_time, test_mse = test_pretraining_network(
        model, criterion, validationloader, device)

    print("{}|{}|{}|{}|{}".format(
        str(i+1).center(7),
        str(train_loss).center(11),
        str(test_loss).center(11),
        str(train_time).center(8),
        str(test_mse).center(7),
    ))

    # save after each epoch since they are long
    torch.save(model.state_dict(),
               os.path.join(MODEL_DIR, f"{i+1}_epochs.pt"))

    if test_loss < best_loss:
        best_loss = test_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if no_improvement == patience:
        print(f"Stopping training at {i+1} epochs -> no improvement in test loss in {patience} epochs")
        break
