import torch
import torch.nn as nn

"""
Discriminator inspired by PatchGAN
"""


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


# we send in input & output imgs together , hence in_channels*2 in initial block
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): #
        super().__init__()
        # initial block
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)  # unpack all stuff of the list

    def forward(self, x, y):
        x = torch.cat([x,y], dim=1)  # along rgb channels
        x = self.initial(x)
        return self.model(x)


def test():
    x = torch.randn((1, 3, 1080, 810))
    y = torch.randn((1, 3, 1080, 810))
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    test()