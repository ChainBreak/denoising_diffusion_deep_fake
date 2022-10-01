#https://github.com/francois-rozet/piqa
from piqa import SSIM
import torch.nn as nn

class MseStructuralSimilarityLoss(nn.Module):
    def __init__(self, input_min_value, input_max_value):
        super().__init__()

        self.input_min_value = input_min_value
        self.input_max_value = input_max_value
        self.ssim = SSIM()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        mse_loss = self.mse(prediction, target)

        prediction = self.normalise_between_zero_and_one(prediction)
        target = self.normalise_between_zero_and_one(target)
        ssim_loss = 1.0 - self.ssim(prediction,target)

        return (mse_loss + ssim_loss) / 2.0

    def normalise_between_zero_and_one(self,x):
        x = (x-self.input_min_value) / (self.input_max_value - self.input_min_value)
        x = x.clip(0.0,1.0)
        return x