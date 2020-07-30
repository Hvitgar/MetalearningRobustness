import abc
import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union
from pathlib import Path
from PIL import Image
import sys
sys.path.append('/gpfs01/bethge/home/bmitzkus/src/stylize-datasets/')
from net import vgg, decoder
from function import adaptive_instance_normalization
from datasets import InfiniteDataLoader, PainterByNumbers
from utils import GumbelSoftmax


# TODO: this should not be here
class TruncatedNormal(nn.Module):
    def __init__(self, a, b, mu, sigma):
        super(TruncatedNormal, self).__init__()
        self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))
        self._sigma_logit = nn.Parameter(torch.tensor(np.log(sigma / (1-sigma)), dtype=torch.float32), requires_grad=True)
        self._mu_logit = nn.Parameter(torch.tensor(np.log(mu / (1-mu)), dtype=torch.float32), requires_grad=True)
        self.register_buffer('zero', torch.tensor(0.))
        self.register_buffer('one', torch.tensor(1.))
        self.eps = torch.tensor(1e-5)
        self.sigmoid = torch.nn.Sigmoid()
        
    @property
    def sigma(self):
        return self.sigmoid(self._sigma_logit)
    
    @property
    def mu(self):
        return self.sigmoid(self._mu_logit)
        
    @property
    def alpha(self):
        return (self.a - self.mu) / self.sigma
    
    @property
    def beta(self):
        return (self.b - self.mu) / self.sigma
    
    @property
    def U(self):
        return torch.distributions.Uniform(self.zero, self.one)
    
    @property
    def N(self):
        return torch.distributions.Normal(self.zero, self.one)
    
    @property
    def Z(self):
        return self.N.cdf(self.beta) - self.N.cdf(self.alpha)
        
    def pdf(self, x):
        x = torch.tensor(x)
        x_norm = (x - self.mu) / self.sigma
        pdf = (1/self.sigma) * (torch.exp(self.N.log_prob(x_norm)) / self.Z)
        pdf = torch.where(self.a <= x, pdf, torch.zeros_like(x))
        pdf = torch.where(x < self.b, pdf, torch.zeros_like(x))
        return pdf
        
    def sample(self, size=None):
        if size is None:
            u = self.U.sample()
        else:
            u = self.U.sample(size)
        sample = self.N.icdf(self.N.cdf(self.alpha) + u * self.Z) * self.sigma + self.mu
        return sample

class Interface(nn.Module, abc.ABC):
    def __init__(self) -> None:
        super(Interface, self).__init__()
        
    @abc.abstractmethod
    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        pass
    
    @abc.abstractmethod
    def _transform(self,
                  img: torch.Tensor) -> torch.Tensor:
        pass
        

class NonParametricTransform(Interface):
    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 4: # batched images
            return self._transform(img)
        elif img.ndim == 3: # single image
            C, H, W = img.size()
            img = img.view(1, C, H, W)
            t_img = self._transform(img)
            return t_img.view(C, H, W)
        else:
            raise ValueError('Found invalid number of dimensions in input image(s)')

class ParametricTransform(Interface):
    def __init__(self, 
                 lower: torch.Tensor, 
                 upper: torch.Tensor,
                 mean: float = 0.05,
                 std: float = 0.01,
                 min_magnitude: float = 0.01,
                 max_magnitude: float = 0.5) -> None:
        super(Interface, self).__init__()
        self.register_buffer('lower', lower)
        self.register_buffer('upper', upper)
        #self.TN = TruncatedNormal(min_magnitude, max_magnitude, mean, std) #TODO fixed aug
        if mean <= 1e-6:
            mean = 1e-6
        elif mean >= 1-1e-6:
            mean = 1-1e-6
        self.mean_logit = nn.Parameter(self.inverse_sigmoid(torch.tensor(mean, dtype=torch.float32)))
        self.sig = nn.Sigmoid()
        
    def forward(self, 
                img: torch.Tensor, 
                magnitude: torch.Tensor = None) -> torch.Tensor:
        if img.ndim == 4: # batched images
            if magnitude is None:
                magnitude = self.sig(self.mean_logit).repeat(img.size()[0])
            return self._transform(img, self.magnitude_to_param(magnitude))
        elif img.ndim == 3: # single image
            C, H, W = img.size()
            img = img.view(1, C, H, W)
            magnitude = self.sig(self.mean_logit)
            t_img = self._transform(img, self.magnitude_to_param(magnitude))
            return t_img.view(C, H, W)
        else:
            raise ValueError('Found invalid number of dimensions in input image(s)')
    
    def magnitude_to_param(self, magnitude: torch.Tensor) -> torch.Tensor:
        return self.lower + (self.upper - self.lower) * magnitude
    
    def inverse_sigmoid(self, value):
        return torch.log(value / (1-value))
    
    @property
    def mu_mag(self):
        #return self.TN.mu #TODO: fixed aug
        return self.sig(self.mean_logit)
                            
    @mu_mag.setter
    def mu_mag(self, value):
        device = self.mu_mag.device
        if value <= 1e-6:
            value = torch.tensor(1e-6, dtype=torch.float32, device=device)
        elif value >= 1-1e-6:
            value = torch.tensor(1-1e-6, dtype=torch.float32, device=device)
        assert 0 < value < 1
        self.mean_logit.data = self.inverse_sigmoid(value)
    
    @property
    def sigma_mag(self):
        #return self.TN.sigma #TODO: fixed aug
        return torch.tensor(0.)
    
    @abc.abstractmethod
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        pass

class SymmetricTransform(ParametricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor(-1, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(SymmetricTransform, self).__init__(lower, upper, **kwargs)
        
    def magnitude_to_param(self,
                          magnitude: torch.Tensor):
        if np.random.uniform() < 0.5:
            return self.lower * magnitude
        else:
            return self.upper * magnitude
        
class Identity(NonParametricTransform):
    def _transform(self,
                  img: torch.Tensor) -> torch.Tensor:
        return img
    
    def __repr__(self):
        return 'Identity'

class Rotate(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor(-180, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(180, dtype=torch.float)
        super(Rotate, self).__init__(lower, upper, **kwargs)
        
    def _transform(self, 
                img: torch.Tensor, 
                param: torch.Tensor) -> torch.Tensor:
        return kornia.rotate(img, param)
    
    def __repr__(self):
        return 'Rotate'
    
class TranslateX(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor([-1, 0], dtype=torch.float)
        if upper is None:
            upper = torch.tensor([1, 0], dtype=torch.float)
        super(TranslateX, self).__init__(lower, upper, **kwargs)
        
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        B, _, _, W = img.shape
        translation = param * W
        return kornia.translate(img, translation.expand(B, 2))
    
    def __repr__(self):
        return 'TranslateX'
    
class TranslateY(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor([0, -1], dtype=torch.float)
        if upper is None:
            upper = torch.tensor([0, 1], dtype=torch.float)
        super(TranslateY, self).__init__(lower, upper, **kwargs)
        
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        B, _, H, _ = img.shape
        translation = param * H
        return kornia.translate(img, translation.expand(B, 2))
    
    def __repr__(self):
        return 'TranslateY'
    
class ShearX(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor([-2, 0], dtype=torch.float)
        if upper is None:
            upper = torch.tensor([2, 0], dtype=torch.float)
        super(ShearX, self).__init__(lower, upper, **kwargs)
        
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = img.shape
        translation = param
        return kornia.shear(img, translation.expand(B, 2))
    
    def __repr__(self):
        return 'ShearX'
    
    
class ShearY(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor([0, -2], dtype=torch.float)
        if upper is None:
            upper = torch.tensor([0, 2], dtype=torch.float)
        super(ShearY, self).__init__(lower, upper, **kwargs)
        
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = img.shape
        translation = param
        return kornia.shear(img, translation.expand(B, 2))
    
    def __repr__(self):
        return 'ShearY'
    
class Brightness(SymmetricTransform):      
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        return kornia.adjust_brightness(img, param)
    
    def __repr__(self):
        return 'Brightness'
    
class Color(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor(-1, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(Color, self).__init__(lower, upper, **kwargs)
    
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        return kornia.adjust_saturation(img, param)
    
    def magnitude_to_param(self,
                          magnitude: torch.Tensor) -> torch.Tensor:
        param = super(Color, self).magnitude_to_param(magnitude)
        return param + 1
    
    def __repr__(self):
        return 'Color'
    
class Contrast(SymmetricTransform):
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor(-1, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(Contrast, self).__init__(lower, upper, **kwargs)
    
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        return kornia.adjust_contrast(img, param)
    
    def magnitude_to_param(self,
                          magnitude: torch.Tensor) -> torch.Tensor:
        param = super(Contrast, self).magnitude_to_param(magnitude)
        return param + 1
    
    def __repr__(self):
        return 'Contrast'
    
class Solarize(ParametricTransform):
    """
        expects images to be in the range [0, 1]
    """
    def __init__(self,
                lower: Union[torch.Tensor, None] = None,
                upper: Union[torch.Tensor, None] = None,
                **kwargs) -> None:
        if lower is None:
            lower = torch.tensor(0, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(Solarize, self).__init__(lower, upper, **kwargs)
    
    def _transform(self,
                  img: torch.Tensor,
                  param: torch.Tensor) -> torch.Tensor:
        return img.where(img < 1-param, 1-img)
    
    def __repr__(self):
        return 'Solarize'
           
    
class Invert(NonParametricTransform):
    """
        expects images to be in the range [0, 1]
    """
    def _transform(self,
                  img: torch.Tensor) -> torch.Tensor:
        return 1 - img
    
    def __repr__(self):
        return 'Invert'
    
    
class StyleTransfer(ParametricTransform):
    def __init__(self, lower=None, upper=None, **kwargs):
        if lower is None:
            lower = torch.tensor(0, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(StyleTransfer, self).__init__(lower, upper, **kwargs)
        
        dec_weights = '/gpfs01/bethge/home/bmitzkus/src/stylize-datasets/models/decoder.pth'
        vgg_weights = '/gpfs01/bethge/home/bmitzkus/src/stylize-datasets/models/vgg_normalised.pth'
        style_dir = '/gpfs01/bethge/data/painter_by_numbers/'
        
        dec = decoder
        enc = vgg
        dec.eval()
        enc.eval()
        dec.load_state_dict(torch.load(dec_weights))
        enc.load_state_dict(torch.load(vgg_weights))
        enc = nn.Sequential(*list(enc.children())[:31])
        self.add_module('vgg', enc)
        self.add_module('decoder', dec)
        self.enc_device = 'cpu'
        self.dec_device = 'cpu'
        
        self.vgg.requires_grad_(False)
        self.decoder.requires_grad_(False)
        
        self._PainterByNumbers = PainterByNumbers(style_dir, transform=transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()]))
        self.resetStyleLoader()
        
    def resetStyleLoader(self, bs=256):
        self.style_loader = iter(InfiniteDataLoader(self._PainterByNumbers, batch_size=bs, num_workers=4, shuffle=True, pin_memory=True, drop_last=True))
        
    def cuda(self, gpu_id):
        super(StyleTransfer, self).cuda(gpu_id)
        self.enc_device = 'cuda:'+str(gpu_id)
        self.dec_device = 'cuda:'+str(gpu_id)
        
    def enc_to(self, target='cuda:0'):
        self.enc_device = target
        self.vgg = self.vgg.to(target)
    
    def dec_to(self, target='cuda:1'):
        self.dec_device = target
        self.decoder = self.decoder.to(target)
        
    def magnitude_to_param(self, magnitude):
        magnitude = magnitude.view([-1, 1, 1, 1])
        alpha = super(StyleTransfer, self).magnitude_to_param(magnitude)
        style = next(self.style_loader)
        return alpha, style
         
    def _transform(self, img, param):
        alpha, style = param
        style = style[:img.size()[0]] # undercomplete batches (last dataloader iteration) or no batching at all
        style = style.to(self.enc_device)
        with torch.no_grad():
            content_f = self.vgg(img)
            style_f = self.vgg(style)
            feat = adaptive_instance_normalization(content_f, style_f)
            
        feat = alpha * feat + (1 - alpha) * content_f
        if self.enc_device != self.dec_device:
            feat = feat.to(self.dec_device)
        stylized = self.decoder(feat)
        if self.enc_device != self.dec_device:
            stylized = stylized.to(self.enc_device)
        return stylized
    
    def __repr__(self):
        return 'StyleTransfer'
    
    
class AdaptiveStyleTransfer(StyleTransfer):
    def __init__(self, temperature=None, logits=None, lower=None, upper=None, **kwargs):
        if lower is None:
            lower = torch.tensor(0, dtype=torch.float)
        if upper is None:
            upper = torch.tensor(1, dtype=torch.float)
        super(AdaptiveStyleTransfer, self).__init__(lower, upper, **kwargs)
        
        
        if temperature is None:
            temperature = torch.tensor(8.)
        if logits is None:
            logits = torch.zeros(128, dtype=torch.float32, requires_grad=True)
            
        self.temperature = temperature
        self.logits = nn.Parameter(logits, requires_grad=True)
        self.G = GumbelSoftmax(self.temperature, logits=self.logits)
        
    def initStyles(self, numStyles, bs=256, seed=None):
        if seed is not None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
        pbn_split, _ = torch.utils.data.dataset.random_split(self._PainterByNumbers, [numStyles, len(self._PainterByNumbers) - numStyles])
        if seed is not None:
            torch.set_rng_state(rng_state)
        
        print('=> Preloading Styles')
        styles = torch.stack([style for style in pbn_split], dim=0)
        print('=> Precomputing style features')
        style_batches = torch.split(styles, bs)
        with torch.no_grad():
            style_features = [self.vgg(batch.to(self.enc_device)) for batch in style_batches]
        self.style_features = torch.cat(style_features).to(self.enc_device)
        print('=> Style Transfer initialized')
        self.individual_alpha = False
        
    def initIndividualAlpha(self):
        self.individual_alpha = True
        mean = self.mu_mag
        self.mean_logit = nn.Parameter(self.inverse_sigmoid(torch.zeros_like(self.logits, dtype=torch.float32) + mean), requires_grad=True)
        
    def magnitude_to_param(self, magnitude):
        if self.individual_alpha:
            bs = len(magnitude) // len(self.mean_logit)
            magnitude = magnitude[:len(self.mean_logit)]
        else:
            bs = magnitude.size()[0]
        style_indices = self.G.rsample([bs], hard=True).to(self.style_features.device) # gumbel softmax sample
        if self.individual_alpha:
            magnitude = torch.einsum("ab,b->a", style_indices, magnitude)
        magnitude = magnitude.view([-1, 1, 1, 1])
        alpha = super(StyleTransfer, self).magnitude_to_param(magnitude)
        style_features = torch.einsum("ab,bcde->acde", style_indices, self.style_features)
        return alpha, style_features
         
    def _transform(self, img, param):
        alpha, style_f = param
        with torch.no_grad():
            content_f = self.vgg(img.to(self.enc_device))
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = alpha * feat + (1 - alpha) * content_f
        feat = feat.to(self.dec_device)
        stylized = self.decoder(feat)
        stylized = stylized.to(self.enc_device)
        return stylized
    
    def __repr__(self):
        return 'AdaptiveStyleTransfer'
    
    
standard_augmentations = [Identity, Rotate, TranslateX, TranslateY, ShearX, ShearY, Brightness, Color, Contrast, Solarize, Invert]