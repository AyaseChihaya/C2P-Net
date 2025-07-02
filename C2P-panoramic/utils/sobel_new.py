import torch
import torch.nn as nn
import numpy as np

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0, bias=False)
        edge_kx = np.array([[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]])
        edge_ky = np.array([[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        self.pad = nn.ReplicationPad2d(1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = self.pad(out)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out
