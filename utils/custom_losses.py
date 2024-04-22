import torch
import torch.nn.functional as F
import torch.nn as nn


# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=3):
#         super(IoULoss, self).__init__()
#         self.classes = n_classes

#     def to_one_hot(self, tensor):
#         # Ensure tensor is of shape (N, H, W)
#         if tensor.dim() == 3:
#             n, h, w = tensor.size()
#         elif tensor.dim() == 4 and tensor.size(1) == 1:  # (N, 1, H, W)
#             tensor = tensor.squeeze(1)  # Remove channel dim
#             n, h, w = tensor.size()
#         else:
#             raise ValueError(f"Expected tensor to be of shape (N, H, W) or (N, 1, H, W), got {tensor.size()}")

#         one_hot = torch.zeros(n, self.classes, h, w, device=tensor.device)
#         one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w).long(), 1)
#         return one_hot

#     def forward(self, inputs, target):
#         # inputs => N x Classes x H x W
#         # target_oneHot => N x Classes x H x W

#         N = inputs.size()[0]

#         # predicted probabilities for each pixel along channel
#         inputs = F.softmax(inputs,dim=1)
        
#         # Numerator Product
#         target_oneHot = self.to_one_hot(target)
#         inter = inputs * target_oneHot
#         ## Sum over all pixels N x C x H x W => N x C
#         inter = inter.view(N,self.classes,-1).sum(2)

#         #Denominator 
#         union= inputs + target_oneHot - (inputs*target_oneHot)
#         ## Sum over all pixels N x C x H x W => N x C
#         union = union.view(N,self.classes,-1).sum(2)

#         loss = inter/union

#         ## Return average loss over classes and batch
#         return 1-loss.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=3):
        super(IoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
        N = inputs.size()[0]
        inputs = F.softmax(inputs, dim=1)
        target_oneHot = F.one_hot(target, num_classes=self.classes).permute(0, 3, 1, 2).float()

        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot - (inputs * target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)

        loss = inter / union
        return 1 - loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=3):
        super(DiceLoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target):
        N = inputs.size()[0]
        inputs = F.softmax(inputs, dim=1)
        target_oneHot = F.one_hot(target, num_classes=self.classes).permute(0, 3, 1, 2).float()

        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)

        union = inputs + target_oneHot
        union = union.view(N, self.classes, -1).sum(2)

        dice = (2. * inter + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=3):
        super(CustomLoss, self).__init__()
        self.n_classes = n_classes
        self.iou_loss = IoULoss(n_classes=n_classes)
        self.dice_loss = DiceLoss(n_classes=n_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')

    def forward(self, inputs, target):
        iou_loss = self.iou_loss(inputs, target)
        dice_loss = self.dice_loss(inputs, target)
        ce_loss = self.ce_loss(inputs, target)

        # Example: simple average of the three losses
        total_loss = (iou_loss + dice_loss + ce_loss) / 3
        return total_loss
