import random, warnings
import torch
import numpy as np

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, *args, shuffle=False, **kwargs):
        sampler = InfiniteSampler(dataset, shuffle)
        super(InfiniteDataLoader, self).__init__(dataset, *args, **kwargs, sampler=sampler)
        
    def __len__(self):
        warnings.warn('You have called __len__() on an infinite iterator.', Warning)
        return 0
    
    
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, shuffle):
        super(InfiniteSampler, self).__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        
    def __iter__(self):
        while True:
            if self.shuffle:
                indices = np.random.permutation(len(self.data_source))
            else:
                indices = range(len(self.data_source))    
            for index in indices:
                yield index
                
    
    def __len__(self):
        warnings.warn('You have called __len__() on an infinite iterator.', Warning)
        return 0