from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import torch
from model import UNet
import numpy as np
import torch.nn as nn
from utils import CocoImages

import time, random
import re
from torch.optim.optimizer import Optimizer, required

def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
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
                    transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
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

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x
      
class UNetContrastive(UNet):
    def __init__(self, n_channels, embedding_dim):
        super().__init__(n_channels, embedding_dim)
        self.projector = ProjectionHead(2048, 256, 128)

    def forward(self, x):
        # Follow the UNet structure until the bottleneck
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = x5.view(x5.size(0), -1)
        
        xp = self.projector(x5)
        return xp
    
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = z_i.size(0) + z_j.size(0)  # Calculate based on actual size
        
        self.mask = self.mask_correlated_samples(N // 2)  # Recreate mask for actual size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        labels = torch.zeros(N, dtype=torch.long).to(positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
EETA_DEFAULT = 0.001

class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True
    
standard_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((256, 256)),
])



COCO_ROOT = "./coco_dataset/raw"
train_dataset = CocoImages(image_root=COCO_ROOT, transform=standard_transform, split="train")
train_dg = CocoDataGen(phase='train', dataset=train_dataset)

val_dataset = CocoImages(image_root=COCO_ROOT, transform=standard_transform, split="val")
val_dg = CocoDataGen(phase='val', dataset=val_dataset)

# Create data loaders
train_loader = DataLoader(train_dg, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, pin_memory_device=device.type)
val_loader = DataLoader(val_dg, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, pin_memory_device=device.type)


model = UNetContrastive(n_channels=3, embedding_dim=256)
model.to(device)
optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)

# "decay the learning rate with the cosine decay schedule without restarts"
#SCHEDULER OR LINEAR EWARMUP
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

#SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#LOSS FUNCTION
criterion = SimCLR_Loss(batch_size = 128, temperature = 0.5)


NUM_EPOCHS = 100
best_loss = float('inf')


nr = 0
tr_loss = []
val_loss = []
set_seed(16)
for epoch in range(NUM_EPOCHS):
    f = open("saved_models/contrastive_learning_new/training_log.txt", "a")
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
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

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
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")
            
            val_loss_epoch += loss.item()
            
    time_taken = (time.time()-stime)/60
    if nr == 0:
        f = open("saved_models/contrastive_learning_new/training_log.txt", "a")
        tr_loss.append(tr_loss_epoch / len(train_loader))
        val_loss.append(val_loss_epoch / len(val_loader))
        f.write(f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
        f.write(f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Validation Loss: {val_loss_epoch / len(val_loader)}\t lr: {round(lr, 5)}")
        f.write(f"\nEpoch [{epoch}/{NUM_EPOCHS}]\t Time Taken: {time_taken} minutes")
        f.close()
        torch.save(model.state_dict(), f"saved_models/contrastive_learning_new/contrastive-learning-{epoch+1}_EPOCHS.pt")

    # Save best model if validation loss improves
    avg_val_loss = val_loss_epoch / len(val_loader)
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), f"saved_models/contrastive_learning_new/lowest_val_model-{epoch+1}_EPOCHS.pt")
        print("Saved new best model")