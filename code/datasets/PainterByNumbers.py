import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
import csv

class PainterByNumbers(VisionDataset):
    def __init__(self, root, transforms=None, transform=None, extensions=None):
        if extensions is None:
            extensions=IMG_EXTENSIONS
        super(PainterByNumbers, self).__init__(root, transform=transform)
        # as make_dataset searches for images in subfolders(=classes) of root, use 'fake class' train
        classes = ['train']
        class_to_idx = {'train': 0}
        samples = datasets.folder.make_dataset(self.root, class_to_idx, extensions)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)
            
            
        self.loader = datasets.folder.default_loader
        self.extensions = extensions
        self.samples = [path for path,_ in samples]
        # filter corrupt files
        corrupted_files = []
        with open('/gpfs01/bethge/home/bmitzkus/Metalearning-Robustness/code/datasets/corrupted_styles.txt', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.samples.remove(row[0])
        
    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return len(self.samples)