import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


n_features_1 = 512  # resnet18-2048, resnet34-2048, resnet50-8192
n_features_2 = 512
fmap_size = 14


def setup_resnet(unfreeze_layers=None):
    resnet = models.resnet34(pretrained=True)
    
    if unfreeze_layers is not None:
        for name, param in resnet.named_parameters():
            layer_name = str(name).split('.')
            if layer_name[0] in unfreeze_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        for param in resnet.parameters():
            param.requires_grad = False
    
    layers = list(resnet.children())[:-2]
    features = nn.Sequential(*layers).cuda()

    return features


def setup_VGG(unfreeze_layers=None):
    vgg = models.vgg16(pretrained=True)

    if unfreeze_layers is not None:
        for name, param in vgg.named_parameters():
            layer_name = str(name).split('.')
            if int(layer_name[1]) >= unfreeze_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        for param in vgg.parameters():
            param.requires_grad = False  # False to freeze all
    
    layers = list(vgg.children())[:-2]
    features = nn.Sequential(*layers).cuda()

    return features


class simple_model(nn.Module):
    def __init__(self, n_classes=100, model='vgg16') -> None:
        super(simple_model, self).__init__()
        self.n_classes = n_classes

        if model == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif model == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif model == 'inception_v3':
            model = models.inception_v3(pretrained=True)
        else:
            raise ValueError('model must be specific')

        for param in model.parameters():
            param.requires_grad = False
        # for name, param in model.named_parameters():
        #     layer_name = str(name).split('.')
        #     if layer_name[0] >= 'layer4':
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        layers = list(model.children())[:-2]
        self.features = nn.Sequential(*layers).cuda()
        
        self.fc = nn.Linear(model.fc.in_features * fmap_size ** 2, n_classes)

        # Initialize the fc
        nn.init.xavier_normal_(self.fc.weight.data)
        
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
    
    def forward(self, x):
        ## X: bs, 3, 448, 448
        ## N = bs
        N = x.size()[0]
        assert x.size() == (N, 3, 448, 448)

        x = self.features(x)
        x = x.view(N, -1)

        x = F.normalize(x)
        x = self.fc(x)

        return x


class BCNN(nn.Module):
    def __init__(self, n_classes=100):
        
        super(BCNN, self).__init__()
        
        self.f1 = setup_VGG(unfreeze_layers=26)
        self.f2 = setup_VGG()

        self.fc = nn.Linear(n_features_1 * n_features_2, n_classes)
        self.dropout = nn.Dropout(0.5)
        
        # Initialize the fc layers.
        nn.init.xavier_normal_(self.fc.weight.data)
        
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
        
    def forward(self, x):
        
        ## X: bs, 3, 448, 448; N = bs
        N = x.size()[0]
        assert x.size() == (N, 3, 448, 448)
        
        ## x : bs, 512, 14, 14
        x_fa = self.f1(x)
        x_fb = self.f2(x)
        assert x_fa.size() == (N, n_features_1, fmap_size, fmap_size)
        
        # bs, 512, 14*2
        x_fa = x_fa.view(N, n_features_1, fmap_size ** 2)
        x_fb = x_fb.view(N, n_features_2, fmap_size ** 2)
        # x_fa = self.dropout(x_fa)
        # x_fb = self.dropout(x_fb)
        assert x_fa.size() == (N, n_features_1, fmap_size ** 2)
        
        # Batch matrix multiplication
        x = torch.bmm(x_fa, torch.transpose(x_fb, 1, 2))/ (fmap_size ** 2) 
        assert x.size() == (N, n_features_1, n_features_2)

        x = x.view(N, n_features_1 * n_features_2)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


if __name__ == '__main__':
    model = BCNN()
    print(model)