import torch
import torch.nn as nn


class RevBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, F, G):
        with torch.no_grad():
            y1 = x1 + F(x2)
            y2 = x2 + G(y1)

        ctx.F = F
        ctx.G = G
        ctx.save_for_backward(y1.detach(), y2.detach())

        return y1, y2

    @staticmethod
    def backward(ctx, dy1, dy2):
        F, G = ctx.F, ctx.G
        y1, y2 = ctx.saved_tensors

        # 2, 3, 4
        with torch.no_grad():
            x2 = y2 - G(y1)
            x1 = y1 - F(x2)

        # 5, 6, 7
        x2.requires_grad_()
        with torch.enable_grad():
            Fx2 = F(x2)
            z1 = x1 + Fx2
            Gz1 = G(z1)

        grads_G = torch.autograd.grad(
            outputs=Gz1,
            inputs=z1,
            grad_outputs=dy2,
            retain_graph=False,
        )

        dz1 = dy1 + grads_G[0]
        grads_G_params = grads_G[1:]

        grads_F = torch.autograd.grad(
            outputs=Fx2,
            inputs=x2,
            grad_outputs=dz1,
            retain_graph=False,
        )

        dx2 = dy2 + grads_F[0]
        grads_F_params = grads_F[1:]
        dx1 = dz1

        # 8, 9
        for p, g in zip(G.parameters(), grads_G_params):
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)

        for p, g in zip(F.parameters(), grads_F_params):
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)

        # in the original paper they also returned the inputs and
        # the updated weights, but this is not necessary as pytorch handles
        # updating the weights by acumulating the gradient in .grad attribute of
        # each parameter. The inputs are not needed as well because of the
        # way the forward() backward() methods of a torch.autograd.Function object work

        # still, this requires 4 return outputs, as per pytorch's documentation
        # https://docs.pytorch.org/docs/stable/generated/torch.autograd.Function.backward.html

        return dx1, dx2, None, None


class RevBottleNeckBlock(nn.Module):
    def __init__(self, channels, width):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=channels),
        )

    def forward(self, x):
        return self.net(x)


class RevBlock(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        assert channels % 2 == 0

        c = channels // 2
        self.F = RevBottleNeckBlock(c, width)
        self.G = RevBottleNeckBlock(c, width)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = RevBlockFunction.apply(x1, x2, self.F, self.G)
        return torch.cat([y1, y2], dim=1)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, width, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class myrevnet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.intro = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, 64, 256, blocks=3)
        self.layer2 = self._make_layer(256, 128, 512, blocks=4)
        self.layer3 = self._make_layer(512, 256, 1024, blocks=6)
        self.layer4 = self._make_layer(1024, 512, 2048, blocks=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_ch, width, out_ch, blocks):
        layers = []

        layers.append(DownsampleBlock(in_ch, width, out_ch))

        for _ in range(blocks - 1):
            layers.append(RevBlock(out_ch, width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.intro(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
