import torchvision
from torchvision.datasets import ImageFolder

class ImageNet16(ImageFolder):
    def __init__(self, root, train=True, **kwargs):
        if train:
            super(ImageNet16, self).__init__(root + 'train', **kwargs)
        else:
            super(ImageNet16, self).__init__(root + 'val', **kwargs)
            
    @classmethod
    def get_normalize_transform(cls):
        return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])