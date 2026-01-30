import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.

        Args:
            alpha: weighting factor for rare class (typically positive class)
            gamma: focusing parameter to down-weight easy examples
            reduction: 'mean', 'sum' or 'none'
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits from model (batch_size, 1) or (batch_size,)
            targets: binary labels (batch_size, 1) or (batch_size,)
        """

        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate probabilities
        probs = torch.sigmoid(inputs)

        # Calculate p_t (probability of true class)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Calculate alpha_t (alpha weighting for true class)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Adaptive Focal Loss that automatically calculates alpha from class frequencies

        Args:
            alpha (list/tensor): If None, will be calculated from data
            gamma (float): Focusing parameter
            reduction (str): Reduction method
        """
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Calculate alpha automatically if not provided
        if self.alpha is None:
            # Calculate class frequencies
            class_counts = torch.bincount(targets, minlength=inputs.size(1)).float()
            total_samples = len(targets)

            # Inverse frequency weighting
            alpha = [total_samples / (2.0 * class_counts[i]) for i in range(len(class_counts))]
            alpha = torch.tensor(alpha).to(inputs.device)
        else:
            alpha = self.alpha

        # Apply alpha weighting
        if alpha is not None:
            if alpha.type() != inputs.data.type():
                alpha = alpha.type_as(inputs.data)
            at = alpha.gather(0, targets.data.view(-1))
            focal_loss = at * (1 - p_t) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
