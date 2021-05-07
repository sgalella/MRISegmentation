import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, *args, **kwargs):
        """UNet for MRI segmentation:

        For more information:
            - https://www.sciencedirect.com/science/article/abs/pii/S0010482519301520?via%3Dihub
        """
        super().__init__(*args, **kwargs)

        self.con_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.con_norm1 = nn.BatchNorm2d(num_features=32)
        self.con_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.con_norm2 = nn.BatchNorm2d(num_features=32)

        self.con_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.con_norm3 = nn.BatchNorm2d(num_features=64)
        self.con_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.con_norm4 = nn.BatchNorm2d(num_features=64)

        self.con_conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.con_norm5 = nn.BatchNorm2d(num_features=128)
        self.con_conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.con_norm6 = nn.BatchNorm2d(num_features=128)

        self.con_conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.con_norm7 = nn.BatchNorm2d(num_features=256)
        self.con_conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.con_norm8 = nn.BatchNorm2d(num_features=256)

        self.con_conv9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.con_norm9 = nn.BatchNorm2d(num_features=512)
        self.con_conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.con_norm10 = nn.BatchNorm2d(num_features=512)

        self.exp_up1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.exp_conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.exp_norm1 = nn.BatchNorm2d(num_features=256)
        self.exp_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.exp_norm2 = nn.BatchNorm2d(num_features=256)

        self.exp_up2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.exp_conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.exp_norm3 = nn.BatchNorm2d(num_features=128)
        self.exp_conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.exp_norm4 = nn.BatchNorm2d(num_features=128)

        self.exp_up3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.exp_conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.exp_norm5 = nn.BatchNorm2d(num_features=64)
        self.exp_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.exp_norm6 = nn.BatchNorm2d(num_features=64)

        self.exp_up4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2)
        self.exp_conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.exp_norm7 = nn.BatchNorm2d(num_features=32)
        self.exp_conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.exp_norm8 = nn.BatchNorm2d(num_features=32)
        self.exp_conv9 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.con_norm1(self.con_conv1(x)))
        copy1 = F.relu(self.con_norm2(self.con_conv2(x)))
        x = F.max_pool2d(copy1, kernel_size=2, stride=2)

        x = F.relu(self.con_norm3(self.con_conv3(x)))
        copy2 = F.relu(self.con_norm4(self.con_conv4(x)))
        x = F.max_pool2d(copy2, kernel_size=2, stride=2)

        x = F.relu(self.con_norm5(self.con_conv5(x)))
        copy3 = F.relu(self.con_norm6(self.con_conv6(x)))
        x = F.max_pool2d(copy3, kernel_size=2, stride=2)

        x = F.relu(self.con_norm7(self.con_conv7(x)))
        copy4 = F.relu(self.con_norm8(self.con_conv8(x)))
        x = F.max_pool2d(copy4, kernel_size=2, stride=2)

        x = F.relu(self.con_norm9(self.con_conv9(x)))
        x = F.relu(self.con_norm10(self.con_conv10(x)))

        x = self.exp_up1(x)
        x = torch.cat((copy4, x), dim=1)
        x = F.relu(self.exp_norm1(self.exp_conv1(x)))
        x = F.relu(self.exp_norm2(self.exp_conv2(x)))

        x = self.exp_up2(x)
        x = torch.cat((copy3, x), dim=1)
        x = F.relu(self.exp_norm3(self.exp_conv3(x)))
        x = F.relu(self.exp_norm4(self.exp_conv4(x)))

        x = self.exp_up3(x)
        x = torch.cat((copy2, x), dim=1)
        x = F.relu(self.exp_norm5(self.exp_conv5(x)))
        x = F.relu(self.exp_norm6(self.exp_conv6(x)))

        x = self.exp_up4(x)
        x = torch.cat((copy1, x), dim=1)
        x = F.relu(self.exp_norm7(self.exp_conv7(x)))
        x = F.relu(self.exp_norm8(self.exp_conv8(x)))
        x = torch.sigmoid(self.exp_conv9(x))

        return x
