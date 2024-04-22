import torch


def IoU(prediction: torch.Tensor, ground_truth: torch.Tensor, classes: list[int] = None) -> list[float]:
    if classes is None:
        classes = torch.unique(ground_truth).tolist()
    ious = torch.zeros(len(classes)).to(prediction.device)
    prediction, ground_truth = prediction.squeeze(), ground_truth.squeeze()
    for i, cls in enumerate(classes):
        pred = prediction == cls
        gt = ground_truth == cls
        intersection = (pred * gt).sum()
        union = (pred | gt).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious[i] = iou
    return ious


def mse(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return ((prediction - ground_truth)**2).mean()

def DiceCoefficient(prediction: torch.Tensor, ground_truth: torch.Tensor, classes: list[int] = None) -> list[float]:
    """
    does 2 * (intersection) / (prediction + gt)
    """
    if classes is None:
        classes = torch.unique(ground_truth).tolist()
    dices = torch.zeros(len(classes)).to(prediction.device)
    prediction, ground_truth = prediction.squeeze(), ground_truth.squeeze()
    for i, cls in enumerate(classes):
        pred = prediction == cls
        gt = ground_truth == cls
        intersection = (pred & gt).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        dices[i] = dice
    return dices

def PixelAccuracy(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    correct = (prediction == ground_truth).sum() # number of correctly predicted pixels
    total = torch.numel(ground_truth) # total num of pixels
    return (correct.float() + 1e-6) / (total + 1e-6)



