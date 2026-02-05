'''
model from training for inference
'''
import torch
import torch.nn as nn
import torchvision.models as models

class MLNet(nn.Module):
    def __init__(self, prior_size):
        super(MLNet, self).__init__()
        # Load vgg16 structure. 
        # Note: We don't need pretrained weights here since we load your checkpoint later, 
        # but using pretrained=True ensures the structure matches exactly if that's what was used.
        features = list(models.vgg16(pretrained=False).features)[:-1]

        # Match specific stride/kernel/padding from the notebook
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2

        self.features = nn.ModuleList(features).eval()
        self.fddropout = nn.Dropout2d(p=0.5)
        self.int_conv = nn.Conv2d(1280, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.prior = nn.Parameter(torch.ones((1, 1, prior_size[0], prior_size[1]), requires_grad=True))
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {16, 23, 29}:
                results.append(x)

        # Concat to get 1280 channels
        x = torch.cat((results[0], results[1], results[2]), 1)
        x = self.fddropout(x)
        x = self.int_conv(x)
        x = self.pre_final_conv(x)

        upscaled_prior = self.bilinearup(self.prior)
        
        # Dot product with prior
        x = x * upscaled_prior
        x = torch.nn.functional.relu(x, inplace=True)
        return x