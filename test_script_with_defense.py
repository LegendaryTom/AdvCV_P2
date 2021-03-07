
from __future__ import print_function
import copy
import  csv
import numpy as np
import os
import numpy
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as nninit
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from vit_with_defense import VisionTransformer

ROOT = '.'

test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True),
                        batch_size=64,
                        shuffle=False,
                        num_workers=4
                        )

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = VisionTransformer(
        img_size=32, patch_size=2, in_chans=3, num_classes=10, embed_dim=80, depth=20,
                 num_heads=20, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    def forward(self,x):
        return self.model(x)

def test(model,test_loader):
    model.eval()
    correct = 0
    avg_act = 0
    for data,target in test_loader:
        data = data.cuda()
        target = target.cuda()
        data16x16 = torch.nn.functional.interpolate(data, size=(16, 16),mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(data)) 
            out16x16 = torch.nn.Softmax(dim=1).cuda()(model(data16x16))
                    
        act,pred = out.max(1, keepdim=True)
        _,pred16x16 = out16x16.max(1, keepdim=True)
        correct += (pred16x16==target.view_as(pred16x16))[pred16x16==pred].sum().cpu()
        avg_act += act.sum().data

    return 100. * float(correct) / len(test_loader.dataset),100. * float(avg_act) / len(test_loader.dataset)


if __name__=="__main__":
        model = NN()
        model.cuda()

        if os.path.isfile("mdl.pth"):
            chk = torch.load("mdl.pth")
            model.load_state_dict(chk["model"]);
            del chk
        torch.cuda.empty_cache();
        acc,_ = test(model,test_loader)
        print('Test accuracy: ',acc)

