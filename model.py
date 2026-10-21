import torch
import torch.nn as nn


class FrameInterpolator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # Training stability
        )
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256))

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1), nn.ReLU())

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, img1, img3):
        input = torch.cat((img1, img3), dim=1)

        # Encoder
        c1 = self.conv1(input)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)

        # Decoder
        u1 = self.up1(c3)
        merge1 = torch.cat((u1, c2), dim=1)  # A skip connection
        d1 = self.dec1(merge1)

        u2 = self.up2(d1)
        merge2 = torch.cat((u2, c1), dim=1)  # A skip connection
        d2 = self.dec2(merge2)

        output = self.final(d2)
        return output
