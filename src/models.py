import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels
import torchvision.transforms as T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Bicubic(nn.Module):

    def __init__ (self, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
    
    def forward(self, inputs):
        # inputs: N * C * Hlow * Wlow
        bicubic_output = F.interpolate(inputs, scale_factor=self.scale_factor, mode='bicubic')
        # bicubic_output: N * C * Hsup * Wsup
        return bicubic_output
    
    
class SRCNN(nn.Module):

    def __init__ (self, scale_factor=2, n_channels=3, f1=9, f2=5, f3=5, h1=64, h2=32):
        super().__init__()

        self.scale_factor = scale_factor

        self.refiner = nn.Sequential(
            nn.Conv2d(n_channels, h1, kernel_size=f1, stride=1, padding=(f1-1) // 2),
            nn.ReLU(),
            nn.Conv2d(h1, h2, kernel_size=f2, stride=1, padding=(f2-1) // 2),
            nn.ReLU(),
            nn.Conv2d(h2, n_channels, kernel_size=f3, stride=1, padding=(f3-1) // 2)
        )
    
    def forward(self, inputs):
        # Bicubic interpolation
        # inputs: N * C * Hlow * Wlow
        bicubic_output = F.interpolate(inputs, scale_factor=self.scale_factor, mode='bicubic')
        # bicubic_output: N * C * Hsup * Wsup

        out = self.refiner(bicubic_output)
        # out: N * C * Hsup * Wsup
        return out
    
    
class SRCNNpp(nn.Module):

    def __init__(self, scale_factor=2, n_channels=3, f_up1=9, f_up2=5, f_ref1=11, f_ref2=5, f_ref3=9, h1=64, h2=32):
        super().__init__()

        self.scale_factor = scale_factor

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_channels, h1, kernel_size=f_up1, stride=1, padding=(f_up1 - 1) // 2),
            nn.Dropout(0.2),
            nn.Conv2d(h1, h2, kernel_size=f_up1, stride=1, padding=(f_up1 - 1) // 2),
            nn.Dropout(0.2),
            nn.Conv2d(h2, n_channels * (scale_factor**2), kernel_size=f_up2, stride=1, padding=(f_up2 - 1) // 2),
            nn.PixelShuffle(scale_factor),
            nn.Sigmoid()
        )
        self.refiner = nn.Sequential(
            nn.Conv2d(n_channels, h1, kernel_size=f_ref1, stride=1, padding=(f_ref1-1) // 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(h1, h2, kernel_size=5, stride=1, padding=(f_ref2-1) // 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(h2, n_channels, kernel_size=f_ref3, stride=1, padding=(f_ref3-1) // 2)
        )

    def forward(self, inputs):
        # inputs: [N, C, Hlow, Wlow]
        out = self.upsampler(inputs)        # out: [N, C, Hsup, Wsup]
        out = self.refiner(out)             # out: [N, C, Hsup, Wsup]
        return out
    
    
class CombinedLoss(nn.Module):

    def __init__(self, pixel_loss_coeff=0.8, n_channels=3):

        super().__init__()

        fe = tvmodels.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(
            fe[0],      # Conv
            fe[1],      # ReLU
            fe[2],      # Conv
            fe[3]       # ReLU
        ).to(DEVICE)
        del fe
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.transformer = T.Compose([
            T.Grayscale(n_channels),
            T.Resize(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.perceptual_loss = 0
        self.pixel_loss = 0
        if 1 >= pixel_loss_coeff >= 0:
            self.pixel_loss_coeff = pixel_loss_coeff  
        else:
            raise ValueError(f"Pixel loss coefficient must be in range [0, 1], got {pixel_loss_coeff} instead")
        self.combined_loss = 0

    def forward(self, model_outputs, targets):
        with torch.no_grad():
            out = self.transformer(model_outputs)
            out = self.feature_extractor(out)
            tar = self.transformer(targets)
            tar = self.feature_extractor(tar)
            self.perceptual_loss = F.mse_loss(out, tar, reduction='mean')
        self.pixel_loss = F.mse_loss(model_outputs, targets)
        self.combined_loss = self.pixel_loss_coeff * self.pixel_loss + (1-self.pixel_loss_coeff) * self.perceptual_loss
        return self.combined_loss