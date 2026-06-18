"""
model.py
CNN architecture from Gamper & Tashev (IWAENC 2018), Table 2 / Fig. 3.

Input:  (batch, 1, 21, 1999) feature matrices (gammatone log-energy)
Output: (batch,) T60 estimate in seconds

Six conv layers (5 filters each, ReLU + BatchNorm), the first four use
1-row kernels (temporal features only); spectral combination happens in
conv5/conv6 and the final fully-connected layer. ~2541 trainable params.
"""

import torch
import torch.nn as nn

# (kernel (F,T), stride (F,T)) per conv layer, from Table 2
_CONV_SPECS = [
    ((1, 10), (1, 2)),
    ((1, 10), (1, 3)),
    ((1, 11), (1, 3)),
    ((1, 11), (1, 2)),
    ((3, 8),  (2, 2)),
    ((4, 7),  (2, 1)),
]
N_FILTERS = 5


class T60CNN(nn.Module):
    def __init__(self, n_bands=21, n_frames=1999):
        super().__init__()
        layers = []
        in_ch = 1
        for kernel, stride in _CONV_SPECS:
            layers += [
                nn.Conv2d(in_ch, N_FILTERS, kernel_size=kernel, stride=stride),
                nn.BatchNorm2d(N_FILTERS),
                nn.ReLU(inplace=True),
            ]
            in_ch = N_FILTERS
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.5)

        # Infer flattened size from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bands, n_frames)
            n_flat = self.conv(dummy).numel()
        self.dense = nn.Linear(n_flat, 1)

    def forward(self, x):
        # x: (B, 1, 21, T)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.flatten(1)
        return self.dense(x).squeeze(-1)


if __name__ == "__main__":
    model = T60CNN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params} (paper: 2541)")

    x = torch.randn(2, 1, 21, 1999)
    model.eval()
    y = model(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}: {y.detach().numpy()}")

    # Trace intermediate shapes (compare to Fig. 3: 994/329/107/49, 10x21, 4x15)
    h = x
    for layer in model.conv:
        h = layer(h)
        if isinstance(layer, torch.nn.Conv2d):
            print(f"  after conv k={layer.kernel_size} s={layer.stride}: "
                  f"{tuple(h.shape[2:])}")
