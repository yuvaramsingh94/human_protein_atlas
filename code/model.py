import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import Normalize

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
        final_out = torch.clamp(final_out, min=0.0, max = 1.0) # add if needed
        #final_out = torch.log(final_out/(1 - final_out))
        return final_out, sigmoid_output

class HpaSub(nn.Module):
    def __init__(self, classes, features):
        super(HpaSub, self).__init__()
        self.species = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, classes),
        )

    def forward(self, GAP):
        #print('GAP ',GAP.shape)
        GAP = F.avg_pool2d(GAP, GAP.size()[2:]).squeeze()
        #rint('GAP ',GAP.shape)
        spe = self.species(GAP)
        return spe

class HpaModel(nn.Module):
    def __init__(self, classes, device, base_model_name, pretrained, features, init_linear_comb = False):
        super(HpaModel, self).__init__()

        mean_list = [0.083170892049318, 0.08627143702844145, 0.05734662013795027, 0.06582942296076659]
        std_list = [0.13561066140407024, 0.13301454127989584, 0.09142918497144226, 0.15651865713966945]
        self.transform=transforms.Compose([Normalize(mean= mean_list,
                              std= std_list,
                              device = device)])
        self.init_linear_comb = init_linear_comb

        base_model = torch.hub.load('pytorch/vision', base_model_name, pretrained=pretrained)
        #print(base_model)
        print('the list ',list(base_model.children()))
        layers = list(base_model.children())[:-1]


        self.init_layer = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1,bias= True)

        if self.init_linear_comb:
            #lets set the weights to combine protein channel with other channel
            #[img_red, img_yellow, img_green, img_blue] # green is protein of interest
            # we will linearly combine green to all the other channel
            #print('yessssssssss')
            custom_weight = torch.tensor([[[1,0,1,0]],[[0,1,1,0]],[[0,0,1,1]]], requires_grad=False).view(3,4,1,1).float()
            self.init_layer.weight = torch.nn.Parameter(custom_weight, requires_grad=False)


        self.model = nn.Sequential(*layers)
        self.fc = HpaSub(classes, features)
        self.autopool = Autopool(input_size = classes, device = device)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            batch_size, cells, C, H, W = x.size()
            c_in = self.transform(x.view(batch_size * cells, C, H, W))
            #print('input c_in ',c_in.shape)
            if self.init_linear_comb:
                c_in = self.init_layer(c_in)
            else:
                c_in = F.relu(self.init_layer(c_in))

            #thinking about adding a batchnorm layer here.........

            #print('init layer c_in ',c_in.shape)
            spe = self.model(c_in)
        spe = self.fc(spe.float())
        spe = spe.contiguous().view(batch_size, cells, -1)
        final_output, sigmoid_output = self.autopool(spe)
        return {'final_output':final_output, 'sigmoid_output':sigmoid_output}
    
    def test(self, x):
        
        batch_size, cells, C, H, W = x.size()
        c_in = self.transform(x.view(batch_size * cells, C, H, W))
        #print('input c_in ',c_in.shape)
        if self.init_linear_comb:
            c_in = self.init_layer(c_in)
        else:
            c_in = F.relu(self.init_layer(c_in))

        #thinking about adding a batchnorm layer here.........

        #print('init layer c_in ',c_in.shape)
        spe = self.fc(self.model(c_in))
        spe = spe.contiguous().view(batch_size, cells, -1)
        final_output, sigmoid_output = self.autopool(spe)
        return {'final_output':final_output, 'sigmoid_output':sigmoid_output}