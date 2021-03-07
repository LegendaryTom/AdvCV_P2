
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

#Added for Attack
from PIL import Image
from advertorch.attacks import PGDAttack
from advertorch.utils import NormalizeByChannelMeanStd
from torchvision.utils import save_image
import time
batch_size=16

ROOT = '.'

test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True),
                        batch_size=batch_size,
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

def attack(model,test_loader):
    model.eval()
    correct = 0
    avg_act = 0

    e = 6
    norm = "inf"
    save_img = 0
    loss = torch.nn.CrossEntropyLoss()

    #Adversary object performs the attack
    adversary = create_adversary(model, e, norm)

    i=0
    for data,target in test_loader:
        data = data.cuda()
        
        target = target.cuda()
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(data))
            prediction = model.forward(data)
                    
        act,pred = out.max(1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().cpu()
        avg_act += act.sum().data

        #Adversarial Attack
        adv_sample = adversary.perturb(data, target)
        adv_sample.cuda()
        adv_prediction = model.forward(adv_sample)

        #Save adversarial images to disk in PNG format so we get discrete pixel values
        saved_adv_sample = torch.empty(batch_size, 3, 32, 32).cuda()
        cloned_adv_sample = adv_sample.clone().cuda()
        for n in range(batch_size):
            image_number = (i*batch_size)+n+1
            image_path = "temp/saved_adv_eps_"+str(e)+"_"+str(image_number)+".PNG"

            save_image(cloned_adv_sample[n], image_path)
            saved_adv_sample[n] = transforms.ToTensor()(Image.open(image_path).convert('RGB'))

            #Saves adv_image and perturbation for report
            #if(save_img == image_number):
            save_img_path = "temp/AdvImage_eps_"+str(e)+"_"+str(image_number)+".PNG"
            save_image(saved_adv_sample[n].clone().detach(), save_img_path)
            
            save_img_path = "temp/Perturbation_eps_"+str(e)+"_"+str(image_number)+".PNG"
            save_image(((data[n].clone().detach()-saved_adv_sample[n].clone().detach())*2)+0.5, save_img_path)
            
            #Print out the original training sample
            #if(e==2):
            save_img_path = "temp/SampleImage_eps_"+str(e)+"_"+str(image_number)+".PNG"
            save_image(data[n].clone().detach(), save_img_path)

        #Predict saved adversarial images
        saved_adv_prediction = model.forward(saved_adv_sample)
        
        #Output Results
        for n in range(batch_size):
            image_number = (i*batch_size)+n+1
            sample_pred = prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on original image
            
            adv_pred = adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on adversarial image
            saved_adv_pred = saved_adv_prediction.argmax(dim=1, keepdim=True)[n].item() #Model's prediction on saved adversarial image
            
            adv_L2_distance = torch.dist(data[n], adv_sample[n], 2).item() #L2 Distance between original and adversarial image
            saved_adv_L2_distance = torch.dist(data[n], saved_adv_sample[n], 2).item() #L2 Distance between original and saved adversarial image
            
            adv_Linf_distance = torch.norm(data[n]-adv_sample[n], float("inf")).item() #Linf Distance between original and adversarial image
            saved_adv_Linf_distance = torch.norm(data[n]-saved_adv_sample[n], float("inf")).item() #Linf Distance between original and saved adversarial image
            
            single_sample_loss = round(loss(prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for original training sample
            single_adv_loss = round(loss(adv_prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for adversarial image
            single_saved_adv_loss = round(loss(saved_adv_prediction[n].unsqueeze(0), target[n].unsqueeze(0)).item(), 6) #Loss for saved adversarial image

            result = [
                "ViT", 
                str(image_number), 
                str(e),
                "L"+str(norm),
                str(target[n].item()), 
                str(sample_pred), 
                str(adv_pred), 
                str(saved_adv_pred), 
                str(round(adv_L2_distance, 6)), 
                str(round(saved_adv_L2_distance, 6)), 
                str(round(adv_Linf_distance, 6)), 
                str(round(saved_adv_Linf_distance, 6)),
                str(single_sample_loss),
                str(single_adv_loss),
                str(single_saved_adv_loss)
            ]
            print(",".join(result))
        i+=1

    return 100. * float(correct) / len(test_loader.dataset),100. * float(avg_act) / len(test_loader.dataset)

def create_adversary(model, epsilon, norm):
    '''
    
    Parameters
    ----------
    model : TYPE
        network model.
    epsilon : number
        max perturbation.
    norm : "2" or "inf"
        Norm used for bounding L_p ball.

    Returns
    -------
    adversary : advertorch adversary
        Adversary object used for performing the attack

    '''
    # Ref: https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.PGDAttack
    # ==============================================================================
    # class advertorch.attacks.PGDAttack(
    #                                   predict, 
    #                                   loss_fn=None, 
    #                                   eps=0.3, 
    #                                   nb_iter=40, 
    #                                   eps_iter=0.01, 
    #                                   rand_init=True, 
    #                                   clip_min=0.0, 
    #                                   clip_max=1.0, 
    #                                   ord=<Mock name='mock.inf' id='140083310782224'>, 
    #                                   l1_sparsity=None, 
    #                                   targeted=False)
    # The projected gradient descent attack (Madry et al, 2017). The attack performs
    # nb_iter steps of size eps_iter, while always staying within eps from the initial
    # point. Paper: https://arxiv.org/pdf/1706.06083.pdf

    # Parameters:   
    #     predict – forward pass function.
    #     loss_fn – loss function.
    #     eps – maximum distortion.
    #     nb_iter – number of iterations.
    #     eps_iter – attack step size.
    #     rand_init – (optional bool) random initialization.
    #     clip_min – mininum value per input dimension.
    #     clip_max – maximum value per input dimension.
    #     ord – (optional) the order of maximum distortion (inf or 2).
    #     targeted – if the attack is targeted.
    
    # ==============================================================================
    # advertorch.attacks.PGDAttack.perturb(self, 
    #                                      x, 
    #                                      y=None)
    # Given examples (x, y), returns their adversarial counterparts with an attack length of eps.

    # Parameters:   
    #     x – input tensor.
    #     y – label tensor. - if None and self.targeted=False, compute y as predicted labels.
    #                       - if self.targeted=True, then y must be the targeted labels.
    # Returns:  
    #     tensor containing perturbed inputs.
    
    iterations = epsilon*2 #From project description
    if(norm=="2" or norm==2):
        L_norm = 2
        step_size = (1.0*epsilon)/iterations
        epsilon_normalized = float(epsilon*32*32) #Multiply by number of pixels, L2 max perturbation also limited to epsilon
    else:
        L_norm = float("inf")
        step_size = 1.0/255.0
        epsilon_normalized = epsilon/255.0 #Normalize to same range as image tensor [0,1]

    adversary = PGDAttack(
                          model, 
                          eps=epsilon_normalized, 
                          eps_iter=step_size, 
                          nb_iter=iterations,
                          rand_init=False, #Maybe this should be true?
                          targeted=False,
                          ord=L_norm
                         )
    return adversary


if __name__=="__main__":
        model = NN()
        model.cuda()

        if os.path.isfile("mdl.pth"):
            chk = torch.load("mdl.pth")
            model.load_state_dict(chk["model"]);
            del chk
        torch.cuda.empty_cache();
        acc,_ = attack(model,test_loader)
        print('Test accuracy: ',acc)

