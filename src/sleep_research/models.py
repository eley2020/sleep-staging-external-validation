from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - gives a clean CLI error before torch install.
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


def require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "PyTorch is required for model training. Install it with the command in "
            "docs/reproduction_notes.md after the local network finishes downloading the wheel."
        ) from TORCH_IMPORT_ERROR


if nn is not None:

    class ResidualBlock2D(nn.Module):
        def __init__(self, channels: int = 64, kernel_size: int = 8) -> None:
            super().__init__()
            padding = (0, kernel_size // 2)
            self.net = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=padding),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=padding),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.net(x)
            if y.shape[-1] > x.shape[-1]:
                y = y[..., : x.shape[-1]]
            elif y.shape[-1] < x.shape[-1]:
                x = x[..., : y.shape[-1]]
            return x + y


    class PaperCNN(nn.Module):
        """PyTorch reconstruction of the reference TF1 CNN feature extractor.

        The TF1 script reshapes each 30 s EEG epoch to [3, 3000, 1] and applies
        2D convolutions over the temporal width: (1, 50) stride 5, max-pool (1, 4),
        (1, 8) stride 1, max-pool (1, 2), nine residual (1, 8) blocks, final
        max-pool (1, 4), width-average, flatten 3 x 64 -> 192, then 288-D features.
        """

        def __init__(self, n_classes: int = 5, feature_dim: int = 288, n_blocks: int = 9) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(1, 50), stride=(1, 5), padding=(0, 25)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                nn.Dropout(p=0.10),
                nn.Conv2d(64, 64, kernel_size=(1, 8), padding=(0, 4)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
            self.blocks = nn.Sequential(*[ResidualBlock2D(64, kernel_size=8) for _ in range(n_blocks)])
            self.tail = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            )
            self.feature = nn.Linear(3 * 64, feature_dim)
            self.dropout = nn.Dropout(p=0.15)
            self.classifier = nn.Linear(feature_dim, n_classes)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                x = x.view(x.shape[0], 3, -1)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            x = self.stem(x)
            x = self.blocks(x)
            x = self.tail(x)
            x = x.mean(dim=-1)
            x = x.flatten(start_dim=1)
            return self.feature(x)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.dropout(self.encode(x)))


    class ContextBiLSTM(nn.Module):
        """Second-stage context model matching the reference 5-epoch BiRNN step."""

        def __init__(
            self,
            feature_dim: int = 288,
            hidden_dim: int = 100,
            n_classes: int = 5,
            dropout: float = 0.50,
            fw_layers: int = 3,
            bw_layers: int = 2,
        ) -> None:
            super().__init__()
            self.input = nn.Linear(feature_dim, hidden_dim)
            self.fw_lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=fw_layers,
                dropout=dropout if fw_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True,
            )
            self.bw_lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=bw_layers,
                dropout=dropout if bw_layers > 1 else 0.0,
                bidirectional=False,
                batch_first=True,
            )
            self.classifier = nn.Linear(hidden_dim * 2, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.input(x))
            fw, _ = self.fw_lstm(x)
            bw_reversed, _ = self.bw_lstm(torch.flip(x, dims=[1]))
            bw = torch.flip(bw_reversed, dims=[1])
            center_index = y_index = x.shape[1] // 2
            center = torch.cat([fw[:, center_index], bw[:, y_index]], dim=1)
            return self.classifier(center)

else:
    ResidualBlock2D = None
    PaperCNN = None
    ContextBiLSTM = None
