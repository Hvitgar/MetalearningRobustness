import kornia
import torch
import torch.nn as nn
import augmentations
from augmentations import ParametricTransform, StyleTransfer


class AugModel(nn.Module):
    def __init__(self, augmentation_mean, augmentation_std, min_magnitude, max_magnitude, augmentations=None, norm_params=None):
        super(AugModel, self).__init__()
        if augmentations is None:
            augmentations = augmentations.standard_augmentations
        self.augmentations = nn.ModuleList(op(mean=augmentation_mean, std=augmentation_std, min_magnitude=min_magnitude, max_magnitude=max_magnitude) if issubclass(op, ParametricTransform) else op() for op in augmentations)
        self.policy_logits = nn.Parameter(torch.zeros(len(augmentations)))
        self.augmentation_iterator = InfiniteIterator(self.augmentations)
        
        if norm_params is None:
            mean = torch.tensor([0, 0, 0])
            std = torch.tensor([1, 1, 1])
        else:
            mean = torch.tensor(norm_params['mean'])
            std = torch.tensor(norm_params['std'])
            
        self.normalize = kornia.color.Normalize(mean=mean, std=std)
              
    def forward(self, x, augmentation=None):
        augmented_images = []
        if self.training:
            if augmentation is not None:
                aug = augmentation
            else:
                aug = next(self.augmentation_iterator)
            if isinstance(aug, StyleTransfer):
                return self.normalize(aug(x))
            for img in x:
                augmented_images.append(aug(img))
        else:
            for img in x:
                policy = torch.distributions.Categorical(logits=self.policy_logits)
                idx = policy.sample()
                aug = self.augmentations[idx]
                augmented_images.append(aug(img))
            
        x = torch.stack(augmented_images, 0)
        x = self.normalize(x)
        return x
    
    
class InfiniteIterator():
    def __init__(self, iterableObject):
        self.iterableObject = iterableObject
        self.iterator = iter(iterableObject)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterableObject)
            item = next(self.iterator)
        return item