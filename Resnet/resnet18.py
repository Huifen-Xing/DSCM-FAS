from copy import deepcopy
import math
import torch.utils.model_zoo as model_zoo
from torch import Tensor
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
normalization = nn.BatchNorm2d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


class Resnet18(nn.Module):
    def __init__(self, pretrained='imagenet',
                 drop_out = 0.5
                ):
        super(Resnet18, self).__init__()
        
        if pretrained == 'imagenet':
            # 加载 ResNet-18 模型
            self.model = resnet18(pretrained=False)  # pretrained
            weights = model_zoo.load_url(model_urls['resnet18'])
            # 加载权重到模型
            self.model.load_state_dict(weights)
            self.model.fc = torch.nn.Identity()  # 将最后一层替换为恒等映射（Identity）

        self.dropout = nn.Dropout(drop_out)


    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x, out_type='feat'):

        # 第一层处理
        x1 = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        x2 = self.model.layer1(x1)
        x3 = self.model.layer2(x2)
        x4 = self.model.layer3(x3)
        x5 = self.model.layer4(x4)
       
        # 平均池化
        avg_x = self.model.avgpool(x5)
        x6 = avg_x
        z = torch.flatten(x6, 1)
        #z = self.dropout(z)

        # 根据输出类型选择返回的结果
        #if out_type == 'ce':
        #    return z, self.fc0(z)
        if out_type == 'feat':
            return z,x4
        else:
            raise RuntimeError("None supported out_type")

    

normalizer = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-10)




