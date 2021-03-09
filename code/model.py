import torch
import torch.nn as nn
import torchvision.models as models

class Autopool(nn.Module):
    def __init__(
        self,
        input_size,
        device,
    ):
        super(Autopool, self).__init__()
        self.alpha = nn.Parameter(requires_grad=True)
        self.alpha.data = torch.ones(
            [input_size], dtype=torch.float32, requires_grad=True, device=device
        )
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        sigmoid_output = self.sigmoid_layer(x)
        alpa_mult_out = torch.mul(sigmoid_output, self.alpha)
        max_tensor = torch.max(alpa_mult_out, dim=1)
        max_tensor_unsqueezed = max_tensor.values.unsqueeze(dim=1)
        softmax_numerator = torch.exp(alpa_mult_out.sub(max_tensor_unsqueezed))
        softmax_den = torch.sum(softmax_numerator, dim=1)
        softmax_den = softmax_den.unsqueeze(dim=1)
        weights = softmax_numerator / softmax_den
        final_out = torch.sum(torch.mul(sigmoid_output, weights), dim=1)
        return final_out, sigmoid_output

class HpaSub(nn.Module):
    def __init__(self, classes):
        super(HpaSub, self).__init__()
        self.species = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, classes),
        )

    def forward(self, GAP):
        spe = self.species(GAP)
        return spe

class HpaModel(nn.Module):
    def __init__(self, classes, device):
        super(HpaModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        del self.model.fc
        self.model.fc = HpaSub(classes)
        self.autopool = Autopool(input_size = classes, device = device)

    def forward(self, x):
        batch_size, cells, C, H, W = x.size()
        c_in = x.view(batch_size * cells, C, H, W)
        spe = self.model(c_in)
        spe = spe.view(batch_size, cells, -1)
        final_output, sigmoid_output = self.autopool(spe)
        return final_output, sigmoid_output