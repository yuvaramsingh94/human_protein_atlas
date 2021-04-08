import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils import Normalize
from efficientnet_pytorch import EfficientNet

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

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)# tanh(self.att(x))
        #norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        #print('norm_att ',norm_att.shape)
        #print('normal ',norm_att)
        #print('normal sum ',torch.sum(norm_att, dim =-1))
        cla = self.nonlinear_transform(self.cla(x))
        #print('cla ',cla.shape)
        x = torch.sum(norm_att * cla, dim=2)
        #print('sum ',x.shape)
        #x = torch.clamp(x, min=0.0, max = 1.0)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

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
    def __init__(self, classes, device, base_model_name, pretrained, features):
        super(HpaModel, self).__init__()
        self.base_model_name = base_model_name
        self.classes = classes
        self.features = features
        mean_list = [0.083170892049318, 0.08627143702844145, 0.05734662013795027, 0.06582942296076659,0.0]
        std_list = [0.13561066140407024, 0.13301454127989584, 0.09142918497144226, 0.15651865713966945,1.]
        self.transform=transforms.Compose([Normalize(mean= mean_list,
                              std= std_list,
                              device = device)])

        if 'efficientnet' in self.base_model_name:
            self.model = EfficientNet.from_pretrained(self.base_model_name)#torch.hub.load('lukemelas/EfficientNet-PyTorch', self.base_model_name, pretrained=pretrained)
            print(self.model)
        else:
            base_model = torch.hub.load('zhanghang1989/ResNeSt', self.base_model_name, pretrained=pretrained) 
            print('the list ',list(base_model.children()))
            layers = list(base_model.children())[:-2]
            self.model = nn.Sequential(*layers)
        self.init_layer = nn.Conv2d(in_channels=5, out_channels=3, kernel_size=1, stride=1,bias= True)
        self.fc1 = nn.Linear(features, features, bias=True)
        self.att_block = AttBlock(features, classes, activation="linear")

        #self.backbone = nn.ModuleList([self.init_layer, self.model])
        #self.fc_attention = nn.ModuleList([self.fc1, self.att_block])
    def init_attention_layer(self,):
        print('hi')
        self.fc1 = nn.Linear(self.features, self.features, bias=True)
        self.att_block = AttBlock(self.features, self.classes, activation="linear")

    def trainable_parameters(self):
        return (list(nn.ModuleList([self.init_layer, self.model]).parameters()), 
                list(nn.ModuleList([self.fc1, self.att_block]).parameters()))
    
    def extract_features(self, x):
        batch_size, cells, C, H, W = x.size()
        c_in = self.transform(x.view(batch_size * cells, C, H, W))
        #print('input c_in ',c_in.shape)
        c_in = F.relu(self.init_layer(c_in))
        #print('init layer c_in ',c_in.shape)
        if 'efficientnet' in self.base_model_name:
            spe = self.model.extract_features(c_in)
        else:
            spe = self.model(c_in)
        spe = F.avg_pool2d(spe, spe.size()[2:]).squeeze()

        return spe.contiguous().view(batch_size, cells, -1)
    
    def attention_section(self,spe):
        spe = F.relu(self.fc1(F.dropout(spe, p=0.5, training=self.training))).permute(0,2,1)
        #print('spe shape ',spe.shape)
        final_output, norm_att, cell_pred = self.att_block(F.dropout(spe, p=0.5, training=self.training))
        cell_pred = torch.sigmoid(cell_pred)
        return {'final_output':final_output, 'cell_pred':cell_pred}


    def forward(self, x):
        batch_size, cells, C, H, W = x.size()
        c_in = self.transform(x.view(batch_size * cells, C, H, W))
        #print('input c_in ',c_in.shape)
        c_in = F.relu(self.init_layer(c_in))
        #print('init layer c_in ',c_in.shape)
        if 'efficientnet' in self.base_model_name:
            spe = self.model.extract_features(c_in)
        else:
            spe = self.model(c_in)
        spe = F.avg_pool2d(spe, spe.size()[2:]).squeeze()
        #print('enc shape ',spe.shape)
        spe = F.relu(self.fc1(F.dropout(spe.contiguous().view(batch_size, cells, -1), p=0.5, training=self.training))).permute(0,2,1)
        #print('spe shape ',spe.shape)
        final_output, norm_att, cell_pred = self.att_block(F.dropout(spe, p=0.5, training=self.training))
        cell_pred = torch.sigmoid(cell_pred)
        return {'final_output':final_output, 'cell_pred':cell_pred}