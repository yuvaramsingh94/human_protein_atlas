import random 
import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.utils.data as data
from sklearn.metrics import f1_score, roc_auc_score
import time
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

class hpa_dataset_v1(data.Dataset):
    def __init__(self, main_df, augmentation = None, path=None,  aug_per = 0.0, cells_used = 8, is_validation = False):
        self.main_df = main_df
        self.aug_percent = aug_per
        self.augmentation = augmentation
        self.label_col = [str(i) for i in range(19)]
        self.cells_used = cells_used
        self.path = path
        self.is_validation = is_validation

        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.main_df)

    def __getitem__(self, idx):
        info = self.main_df.iloc[idx]
        ids = info["ID"]

        target_vec = info[self.label_col].values.astype(np.int)

        # lets begin
        cell_count = len(os.listdir(os.path.join(self.path,f'{ids}')))
        
        if not self.is_validation:

            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample

                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            #print('augmentation')
                            vv = self.augmentation(image= vv)["image"]
                        cell_list.append(vv)
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                #print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        vv = h['train_img'][...]
                        if random.random() < self.aug_percent:
                            vv = self.augmentation(image= vv)["image"]
                        cell_list.append(vv)
                train_img = np.array(cell_list)
                shape = (self.cells_used - cell_count + 1, 224, 224, 4)
                zero_arr = np.zeros(shape, dtype=float)
                #print('zero_arr ',zero_arr.shape)
                #print('train_img ',train_img.shape)
                train_img = np.concatenate([train_img, zero_arr], axis=0)
                target_vec[-1] = 1# as we are adding black img . negative = 1 also
        else:
            if cell_count == self.cells_used:
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)
                        
            elif cell_count > self.cells_used:#random downsample
                
                cell_list = []
                for i in range(1, self.cells_used + 1):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)

            elif cell_count < self.cells_used:# add zero images
                #print('in the less class')
                cell_list = []
                for i in range(1, cell_count):
                    hdf5_path = os.path.join(self.path,ids,f'{ids}_{i}.hdf5')
                    with h5py.File(hdf5_path,"r") as h:
                        cell_list.append(h['train_img'][...])
                train_img = np.array(cell_list)
                shape = (self.cells_used - cell_count + 1, 224, 224, 4)
                zero_arr = np.zeros(shape, dtype=float)
                #print('zero_arr ',zero_arr.shape)
                #print('train_img ',train_img.shape)
                train_img = np.concatenate([train_img, zero_arr], axis=0)
                target_vec[-1] = 1# as we are adding black img . negative = 1 also
        #print('this is the shape ', train_img.shape)
        #print("{} seconds".format(end_time-start_time))
        #return {'image' : torch.from_numpy(train_img), 'label' : torch.from_numpy(target_vec)}
        return {'image' : train_img, 'label' : target_vec}

def score_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ROC_AUC_score = roc_auc_score(labels, preds, average='micro')
    preds = preds > 0.5
    F1_score = f1_score(labels, preds, average='micro')
    

    return {'AUROC':ROC_AUC_score,'F1_score':F1_score}

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, if_sigmoid = False, device = None):
        super(focal_loss, self).__init__()
        self.alpha = torch.Tensor([alpha, 1 - alpha])  # .cuda()
        self.alpha = self.alpha.to(device)
        self.gamma = gamma
        self.if_sigmoid = if_sigmoid
        #self.bce_loss = nn.BCELoss(reduction="none")

    def forward(
        self,
        inputs,
        targets,
    ):
        if self.if_sigmoid:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction= 'none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction= 'none')
        targets = targets.type(torch.long)
        ## i guess this is going to give use some thing like [0.25,0.25,1-0.25]
        ## for target of [0,0,1]. i guess gather will do this for us
        at = self.alpha.gather(0, targets.data.view(-1))
        ## here we apply the exp for the log values
        ## this is very tricky. if you see this is for choosing p pr 1-p based on 0 or 1
        ## its an inteligent way to do the choosing and araiving at the value fast
        ## without this you have to do some hard engineering to get this value
        #print('bce ',BCE_loss.shape)
        
        pt = torch.exp(-BCE_loss)
        #print('rest ',(at * (1.0 - pt) ** self.gamma))

        F_loss = (at * (1.0 - pt) ** self.gamma) * (BCE_loss)
        return F_loss.mean()

class PANNsLoss(nn.Module):
    def __init__(self, device = None):
        super().__init__()

        self.bce = focal_loss(alpha=0.3, gamma=3, if_sigmoid = False , device = device)#nn.BCELoss(weight = torch.tensor(class_weights, requires_grad = False))

    def forward(self, input, target):
        input_ = input#["final_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()
        #print(input_.shape)
        return self.bce(input_.view(-1), target.view(-1))

def lsep_loss_stable(input, target, average=True):

    n = input.size(0)

    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_lower = (target.unsqueeze(1) < target.unsqueeze(2)).float()

    differences = differences.view(n, -1)
    where_lower = where_lower.view(n, -1)

    max_difference, index = torch.max(differences, dim=1, keepdim=True)
    differences = differences - max_difference
    exps = differences.exp() * where_lower

    lsep = max_difference + torch.log(torch.exp(-max_difference) + exps.sum(-1))

    if average:
        return lsep.mean()
    else:
        return lsep



class ImprovedPANNsLoss(nn.Module):
    def __init__(self, output_key="logit", weights=[1, 0.5]):
        super().__init__()

        self.output_key = output_key
        if output_key == "logit":
            self.normal_loss = nn.BCEWithLogitsLoss()
        else:
            self.normal_loss = nn.BCELoss()

        self.bce = nn.BCELoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target.float()

        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        normal_loss = self.normal_loss(input_, target)
        auxiliary_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss
'''
class ImprovedPANNsLoss(nn.Module):
    def __init__(self, output_key="logit", weights=[1, 0.5]):
        super().__init__()
        self.output_key = output_key
        if output_key == "logit":
            self.normal_loss = focal_loss(alpha=0.3, gamma=3,if_sigmoid = True)#nn.BCEWithLogitsLoss()
        else:
            self.normal_loss = focal_loss(alpha=0.3, gamma=3,if_sigmoid = False)#nn.BCELoss()
        self.bce = focal_loss(alpha=0.3, gamma=3,if_sigmoid = False)#nn.BCELoss()
        self.weights = weights
    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target.float()
        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)
        normal_loss = self.normal_loss(input_.view(-1), target.view(-1))
        auxiliary_loss = self.bce(clipwise_output_with_max.view(-1), target.view(-1))
        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss
'''







