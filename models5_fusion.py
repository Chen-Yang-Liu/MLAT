import torch
from torch import nn
import torchvision
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda:0


class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    """

    def __init__(self, NetType='resnet50', encoded_image_size=14, attention_method="ByPixel"):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.attention_method = attention_method

        # resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        net = torchvision.models.inception_v3(pretrained=True, transform_input=False) if NetType == 'inception_v3' else \
              torchvision.models.vgg16(pretrained=True) if NetType == 'vgg16' else \
              torchvision.models.resnet50(pretrained=True) if NetType == 'resnet50' else torchvision.models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        # Specifically, Remove: AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=2048, out_features=1000, bias=True)]

        # modules = list(net.children())[:-2]
        modules = list(net.children())[:-1] if NetType == 'inception_v3' or NetType == 'vgg16' else list(net.children())[:-2]
        # modules = list(net.children())[:-1] if NetType == 'inception_v3' else list(net.children())[:-2]  # -2 for resnet & vgg
        if NetType == 'inception_v3': del modules[13]

        self.net = nn.Sequential(*modules)

        # every block of resnet for fusion
        if NetType == 'resnet50' or NetType == 'resnet101' or NetType == 'resnet152':
            resnet_block1 = list(net.children())[:5]
            self.resnet_block1 = nn.Sequential(*resnet_block1)
            resnet_block2 = list(net.children())[5]
            self.resnet_block2 = nn.Sequential(*resnet_block2)
            resnet_block3 = list(net.children())[6]
            self.resnet_block3 = nn.Sequential(*resnet_block3)
            resnet_block4 = list(net.children())[7]
            self.resnet_block4 = nn.Sequential(*resnet_block4)
            self.conv4 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1)
            self.conv3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
            self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)

        # if self.attention_method == "ByChannel":
        #     self.cnn1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #     self.bn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     self.relu = nn.ReLU(inplace=True)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # self.adaptive_pool4 = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # self.adaptive_pool3 = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        # with fusion for resnet
        out1 = self.resnet_block1(images)  # 256
        out2 = self.resnet_block2(out1)  # 512
        out3 = self.resnet_block3(out2)  # 1024
        out4 = self.resnet_block4(out3)  # 2048

        # # FIXME:concat432
        p4 = self.conv4(out4)  # 1024
        p3 = self.conv3(out3)  # 512
        p2 = self.conv2(out2)  # 512
        out = torch.cat([F.interpolate(p4, scale_factor=2), p3, F.interpolate(p2, scale_factor=0.5)], dim=1)

        # without fusion
        # out = self.net(images)  # (batch_size, 2048, image_size/32, image_size/32)
        # if self.attention_method == "ByChannel":
        #     out = self.relu(self.bn1(self.cnn1(out)))
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14] #FIXME:for fusion
        out = out.permute(0, 2, 3, 1)
        return out



    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.net.children())[5:]:  # FIXME:maybe try 6:
            for p in c.parameters():
                p.requires_grad = fine_tune

