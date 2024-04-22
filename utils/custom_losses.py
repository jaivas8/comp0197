import torch
import torch.nn.functional as F
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=3):
#         super(IoULoss, self).__init__()
#         self.classes = n_classes

#     def forward(self, inputs, target):
#         N = inputs.size()[0]
#         inputs = F.softmax(inputs, dim=1)
#         target_oneHot = F.one_hot(
#             target, num_classes=self.classes).permute(0, 3, 1, 2).float()

#         inter = inputs * target_oneHot
#         inter = inter.view(N, self.classes, -1).sum(2)

#         union = inputs + target_oneHot - (inputs * target_oneHot)
#         union = union.view(N, self.classes, -1).sum(2)

#         loss = inter / union
#         return 1 - loss.mean()


# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=3):
#         super(DiceLoss, self).__init__()
#         self.classes = n_classes

#     def forward(self, inputs, target):
#         N = inputs.size()[0]
#         inputs = F.softmax(inputs, dim=1)
#         target_oneHot = F.one_hot(
#             target, num_classes=self.classes).permute(0, 3, 1, 2).float()

#         inter = inputs * target_oneHot
#         inter = inter.view(N, self.classes, -1).sum(2)

#         union = inputs + target_oneHot
#         union = union.view(N, self.classes, -1).sum(2)

#         dice = (2. * inter + 1e-6) / (union + 1e-6)
#         return 1 - dice.mean()


# class CustomLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True, n_classes=3):
#         super(CustomLoss, self).__init__()
#         self.n_classes = n_classes
#         self.iou_loss = IoULoss(n_classes=n_classes)
#         self.dice_loss = DiceLoss(n_classes=n_classes)
#         self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')

#     def forward(self, inputs, target):
#         iou_loss = self.iou_loss(inputs, target)
#         dice_loss = self.dice_loss(inputs, target)
#         ce_loss = self.ce_loss(inputs, target)

#         # Example: simple average of the three losses
#         total_loss = (iou_loss + dice_loss + ce_loss) / 3
#         return total_loss


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

        self.mask = self.mask_correlated_samples(
            N // 2)  # Recreate mask for actual size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(
            1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, N // 2)
        sim_j_i = torch.diag(sim, -N // 2)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N, dtype=torch.long).to(positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
