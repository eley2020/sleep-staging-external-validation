import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy


########################################################################################


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECAFeatureBlock(nn.Module):
    """ECA replacement for the AFR residual block."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=3):
        super(ECAFeatureBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.eca = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        y = out.mean(dim=-1, keepdim=True).transpose(1, 2)
        y = torch.sigmoid(self.eca(y)).transpose(1, 2)
        out = out * y
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class CBAMFeatureBlock(nn.Module):
    """CBAM-style channel and temporal attention replacement for AFR."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 temporal_kernel_size=7):
        super(CBAMFeatureBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        hidden = max(1, planes // 16)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(planes, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, planes, kernel_size=1, bias=False),
        )
        self.temporal_attention = nn.Conv1d(
            2, 1, kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2, bias=False,
        )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        avg = out.mean(dim=-1, keepdim=True)
        mx = out.amax(dim=-1, keepdim=True)
        channel_weights = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))
        out = out * channel_weights
        temporal = torch.cat(
            (out.mean(dim=1, keepdim=True), out.amax(dim=1, keepdim=True)), dim=1
        )
        temporal_weights = torch.sigmoid(self.temporal_attention(temporal))
        out = out * temporal_weights
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class P2ResidualTemporalBlock(nn.Module):
    """Small identity-initialized residual refinement for the long branch."""

    def __init__(self, channels=128, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.act = GELU()
        # Zero residual strength makes the initial function exactly unchanged.
        self.raw_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        update = self.pointwise(self.act(self.bn(self.depthwise(x))))
        scale = 0.1 * torch.tanh(self.raw_scale)
        return x + scale * update


class PIDTemporalStem(nn.Module):
    """Identity-initialized multi-order temporal input refinement.

    ``integral`` is a local RMS envelope and ``derivative`` is a smoothed
    central difference.  Each fixed transform is RMS-matched to the original
    signal before a tiny depthwise refinement.  Zero-initialized per-channel
    residual gates make every mode exactly equivalent to the unmodified input
    at initialization, which permits fair warm starts from existing models.
    """

    MODES = ("none", "integral", "derivative", "pid", "matched")

    def __init__(
        self,
        channels,
        mode="none",
        integral_kernel=101,
        derivative_smooth_kernel=5,
        refinement_kernel=9,
        max_residual_scale=0.5,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unsupported PID temporal mode: {}".format(mode))
        for name, value in (
            ("integral_kernel", integral_kernel),
            ("derivative_smooth_kernel", derivative_smooth_kernel),
            ("refinement_kernel", refinement_kernel),
        ):
            if int(value) <= 0 or int(value) % 2 == 0:
                raise ValueError("{} must be a positive odd integer".format(name))
        if max_residual_scale <= 0.0:
            raise ValueError("max_residual_scale must be positive")
        self.channels = int(channels)
        self.mode = str(mode)
        self.integral_kernel = int(integral_kernel)
        self.derivative_smooth_kernel = int(derivative_smooth_kernel)
        self.max_residual_scale = float(max_residual_scale)
        branch_names = {
            "none": (),
            "integral": ("integral",),
            "derivative": ("derivative",),
            "pid": ("integral", "derivative"),
            # Parameter-matched control: two ordinary residual refinements of
            # the raw input, without integral/derivative transforms.
            "matched": ("raw1", "raw2"),
        }[self.mode]
        self.branch_names = branch_names
        self.refinements = nn.ModuleDict()
        self.raw_scales = nn.ParameterDict()
        for branch_name in branch_names:
            conv = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=int(refinement_kernel),
                padding=int(refinement_kernel) // 2,
                groups=self.channels,
                bias=False,
            )
            nn.init.zeros_(conv.weight)
            center = int(refinement_kernel) // 2
            with torch.no_grad():
                conv.weight[:, 0, center] = 1.0
            self.refinements[branch_name] = conv
            self.raw_scales[branch_name] = nn.Parameter(torch.zeros(self.channels))
        self.last_scales = None

    @staticmethod
    def _same_avg_pool(x, kernel_size):
        return F.avg_pool1d(
            x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
            count_include_pad=False,
        )

    def _integral_transform(self, x):
        # A local energy accumulation is more stable for band-passed PSG than
        # an unbounded cumulative sum, which is dominated by baseline drift.
        return torch.sqrt(self._same_avg_pool(x.square(), self.integral_kernel).clamp_min(1e-18))

    def _derivative_transform(self, x):
        smooth = self._same_avg_pool(x, self.derivative_smooth_kernel)
        padded = F.pad(smooth, (1, 1), mode="replicate")
        return 0.5 * (padded[..., 2:] - padded[..., :-2])

    @staticmethod
    def _rms_match(branch, reference):
        # Remove the branch DC component, then preserve the reference channel
        # scale so residual fusion remains meaningful in physical volts.
        branch = branch - branch.mean(dim=-1, keepdim=True)
        branch_rms = torch.sqrt(branch.square().mean(dim=-1, keepdim=True).clamp_min(1e-18))
        reference_rms = torch.sqrt(reference.square().mean(dim=-1, keepdim=True).clamp_min(1e-18))
        return branch * (reference_rms / branch_rms)

    def forward(self, x):
        if self.mode == "none":
            self.last_scales = {}
            return x
        transformed = {
            "integral": lambda: self._integral_transform(x),
            "derivative": lambda: self._derivative_transform(x),
            "raw1": lambda: x,
            "raw2": lambda: x,
        }
        out = x
        scales = {}
        for branch_name in self.branch_names:
            branch = self._rms_match(transformed[branch_name](), x)
            branch = self.refinements[branch_name](branch)
            scale = self.max_residual_scale * torch.tanh(
                self.raw_scales[branch_name]
            )
            out = out + scale.view(1, -1, 1) * branch
            scales[branch_name] = scale.detach()
        self.last_scales = scales
        return out


class DynamicMultiScalePIDTemporalStem(nn.Module):
    """Dynamic, modality-aware, multi-scale PID-inspired refinement.

    Integral and derivative scales are mixed independently for every channel.
    A shared gate MLP consumes per-epoch channel statistics plus a learned
    channel embedding and produces sample-specific I/D residual strengths.
    Its final layer is zero-initialized, so the complete module begins as an
    exact identity map despite the stronger dynamic parameterization.
    """

    MODES = (
        "dynamic_integral",
        "dynamic_derivative",
        "dynamic_pid",
        "dynamic_matched",
    )

    def __init__(
        self,
        channels,
        mode="dynamic_pid",
        integral_kernels=(51, 101, 201, 401),
        derivative_smooth_kernels=(3, 5, 11, 21),
        refinement_kernel=9,
        channel_embedding_dim=4,
        gate_hidden_size=16,
        max_residual_scale=0.5,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unsupported dynamic PID temporal mode: {}".format(mode))
        integral_kernels = tuple(int(v) for v in integral_kernels)
        derivative_smooth_kernels = tuple(
            int(v) for v in derivative_smooth_kernels
        )
        for name, values in (
            ("integral_kernels", integral_kernels),
            ("derivative_smooth_kernels", derivative_smooth_kernels),
        ):
            if not values or any(value <= 0 or value % 2 == 0 for value in values):
                raise ValueError("{} must contain positive odd integers".format(name))
        if refinement_kernel <= 0 or refinement_kernel % 2 == 0:
            raise ValueError("refinement_kernel must be a positive odd integer")
        if channel_embedding_dim <= 0 or gate_hidden_size <= 0:
            raise ValueError("dynamic PID gate dimensions must be positive")
        if max_residual_scale <= 0.0:
            raise ValueError("max_residual_scale must be positive")
        self.channels = int(channels)
        self.mode = str(mode)
        self.integral_kernels = integral_kernels
        self.derivative_smooth_kernels = derivative_smooth_kernels
        self.max_residual_scale = float(max_residual_scale)
        self.branch_names = {
            "dynamic_integral": ("integral",),
            "dynamic_derivative": ("derivative",),
            "dynamic_pid": ("integral", "derivative"),
            "dynamic_matched": ("raw1", "raw2"),
        }[self.mode]
        # Per-channel scale selection makes the temporal receptive field
        # modality-specific while retaining the same candidate windows.
        self.integral_scale_logits = nn.Parameter(
            torch.zeros(self.channels, len(self.integral_kernels))
        )
        self.derivative_scale_logits = nn.Parameter(
            torch.zeros(self.channels, len(self.derivative_smooth_kernels))
        )
        self.refinements = nn.ModuleDict()
        for branch_name in self.branch_names:
            conv = nn.Conv1d(
                self.channels,
                self.channels,
                kernel_size=int(refinement_kernel),
                padding=int(refinement_kernel) // 2,
                groups=self.channels,
                bias=False,
            )
            nn.init.zeros_(conv.weight)
            with torch.no_grad():
                conv.weight[:, 0, int(refinement_kernel) // 2] = 1.0
            self.refinements[branch_name] = conv
        self.channel_embedding = nn.Parameter(
            torch.zeros(self.channels, int(channel_embedding_dim))
        )
        self.gate_network = nn.Sequential(
            nn.Linear(5 + int(channel_embedding_dim), int(gate_hidden_size)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_size), 2),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.zeros_(self.gate_network[-1].bias)
        self.last_scales = None
        self.last_integral_scale_weights = None
        self.last_derivative_scale_weights = None

    @staticmethod
    def _same_avg_pool(x, kernel_size):
        return F.avg_pool1d(
            x, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, count_include_pad=False,
        )

    @staticmethod
    def _rms_match(branch, reference):
        branch = branch - branch.mean(dim=-1, keepdim=True)
        branch_rms = torch.sqrt(
            branch.square().mean(dim=-1, keepdim=True).clamp_min(1e-18)
        )
        reference_rms = torch.sqrt(
            reference.square().mean(dim=-1, keepdim=True).clamp_min(1e-18)
        )
        return branch * (reference_rms / branch_rms)

    def _integral_transform(self, x):
        candidates = torch.stack(
            [
                torch.sqrt(
                    self._same_avg_pool(x.square(), kernel).clamp_min(1e-18)
                )
                for kernel in self.integral_kernels
            ],
            dim=2,
        )
        weights = torch.softmax(self.integral_scale_logits, dim=-1)
        self.last_integral_scale_weights = weights.detach()
        return torch.sum(candidates * weights[None, :, :, None], dim=2)

    def _derivative_transform(self, x):
        candidates = []
        for kernel in self.derivative_smooth_kernels:
            smooth = self._same_avg_pool(x, kernel)
            padded = F.pad(smooth, (1, 1), mode="replicate")
            candidates.append(0.5 * (padded[..., 2:] - padded[..., :-2]))
        candidates = torch.stack(candidates, dim=2)
        weights = torch.softmax(self.derivative_scale_logits, dim=-1)
        self.last_derivative_scale_weights = weights.detach()
        return torch.sum(candidates * weights[None, :, :, None], dim=2)

    def _dynamic_scales(self, x, integral, derivative):
        rms_uv = torch.sqrt(x.square().mean(dim=-1).clamp_min(1e-18)) * 1e6
        diff = x[..., 1:] - x[..., :-1]
        diff_rms_uv = torch.sqrt(
            diff.square().mean(dim=-1).clamp_min(1e-18)
        ) * 1e6
        mean_abs_uv = x.abs().mean(dim=-1) * 1e6
        integral_rms_uv = torch.sqrt(
            integral.square().mean(dim=-1).clamp_min(1e-18)
        ) * 1e6
        derivative_rms_uv = torch.sqrt(
            derivative.square().mean(dim=-1).clamp_min(1e-18)
        ) * 1e6
        features = torch.stack(
            (
                rms_uv,
                diff_rms_uv,
                mean_abs_uv,
                integral_rms_uv,
                derivative_rms_uv,
            ),
            dim=-1,
        )
        features = torch.log1p(features)
        embeddings = self.channel_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        logits = self.gate_network(torch.cat((features, embeddings), dim=-1))
        return self.max_residual_scale * torch.tanh(logits)

    def forward(self, x):
        integral = self._integral_transform(x)
        derivative = self._derivative_transform(x)
        dynamic_scales = self._dynamic_scales(x, integral, derivative)
        transformed = {
            "integral": integral,
            "derivative": derivative,
            "raw1": x,
            "raw2": x,
        }
        scale_indices = {
            "integral": 0,
            "raw1": 0,
            "derivative": 1,
            "raw2": 1,
        }
        out = x
        scales = {}
        for branch_name in self.branch_names:
            branch = self._rms_match(transformed[branch_name], x)
            branch = self.refinements[branch_name](branch)
            scale = dynamic_scales[..., scale_indices[branch_name]]
            out = out + scale.unsqueeze(-1) * branch
            scales[branch_name] = scale.detach()
        self.last_scales = scales
        return out
        
        
class MRCNN(nn.Module):
    def __init__(
        self,
        afr_reduced_cnn_size,
        input_channels=1,
        afr_type="se",
        modal_stem=False,
        modality_groups=None,
        cross_modal_attention=False,
        film_modulation=False,
        branch_calibration=False,
        dense_long_branch=False,
        channelwise_branch_calibration=False,
        p2_residual=False,
        afr_bypass=False,
        post_p3_mode="full",
        pid_temporal_mode="none",
        pid_integral_kernel=101,
        pid_derivative_smooth_kernel=5,
        pid_refinement_kernel=9,
        pid_max_residual_scale=0.5,
        pid_integral_kernels=(51, 101, 201, 401),
        pid_derivative_smooth_kernels=(3, 5, 11, 21),
        pid_dynamic_channel_embedding_dim=4,
        pid_dynamic_gate_hidden_size=16,
    ):
        super(MRCNN, self).__init__()
        self.modal_stem = bool(modal_stem)
        self.modality_groups = None
        self.cross_modal_attention = bool(cross_modal_attention)
        self.film_modulation = bool(film_modulation)
        self.branch_calibration = bool(branch_calibration)
        self.dense_long_branch = bool(dense_long_branch)
        self.channelwise_branch_calibration = bool(channelwise_branch_calibration)
        self.p2_residual = bool(p2_residual)
        self.afr_bypass = bool(afr_bypass)
        self.post_p3_mode = str(post_p3_mode)
        dynamic_pid_modes = DynamicMultiScalePIDTemporalStem.MODES
        if pid_temporal_mode in dynamic_pid_modes:
            self.pid_temporal_stem = DynamicMultiScalePIDTemporalStem(
                input_channels,
                mode=pid_temporal_mode,
                integral_kernels=pid_integral_kernels,
                derivative_smooth_kernels=pid_derivative_smooth_kernels,
                refinement_kernel=pid_refinement_kernel,
                channel_embedding_dim=pid_dynamic_channel_embedding_dim,
                gate_hidden_size=pid_dynamic_gate_hidden_size,
                max_residual_scale=pid_max_residual_scale,
            )
        else:
            self.pid_temporal_stem = PIDTemporalStem(
                input_channels,
                mode=pid_temporal_mode,
                integral_kernel=pid_integral_kernel,
                derivative_smooth_kernel=pid_derivative_smooth_kernel,
                refinement_kernel=pid_refinement_kernel,
                max_residual_scale=pid_max_residual_scale,
            )
        if self.channelwise_branch_calibration and not self.branch_calibration:
            raise ValueError("channelwise branch calibration requires --branch-calibration")
        if self.film_modulation and not self.modal_stem:
            raise ValueError("FiLM modulation requires --modal-stem")
        if self.modal_stem:
            if modality_groups is None:
                if self.film_modulation:
                    raise ValueError("FiLM modulation requires explicit modality groups")
                self.modal_stem_conv = nn.Conv1d(
                    input_channels, input_channels, kernel_size=7, stride=1,
                    padding=3, groups=input_channels, bias=False,
                )
                self.modal_stem_bn = nn.BatchNorm1d(input_channels)
                self.modal_stem_groups = None
            else:
                groups = [tuple(int(i) for i in group) for group in modality_groups]
                flat = [i for group in groups for i in group]
                if sorted(flat) != list(range(input_channels)):
                    raise ValueError(
                        "modality_groups must partition input channels exactly; "
                        f"got {groups} for input_channels={input_channels}"
                    )
                self.modality_groups = groups
                self.modal_stem_groups = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(len(group), len(group), kernel_size=7,
                                      padding=3, groups=len(group), bias=False),
                            nn.BatchNorm1d(len(group)),
                            GELU(),
                        )
                        for group in groups
                    ]
                )
                if self.cross_modal_attention:
                    if len(groups) < 2:
                        raise ValueError("cross-modal attention requires at least two modality groups")
                    token_dim = 16
                    self.cross_modal_proj = nn.ModuleList(
                        [nn.Conv1d(len(group), token_dim, kernel_size=1, bias=False) for group in groups]
                    )
                    self.cross_modal_attn = nn.MultiheadAttention(
                        embed_dim=token_dim, num_heads=4, dropout=0.1, batch_first=True
                    )
                    self.cross_modal_norm = nn.LayerNorm(token_dim)
                    self.cross_modal_out = nn.ModuleList(
                        [nn.Conv1d(token_dim, len(group), kernel_size=1, bias=False) for group in groups]
                    )
                if self.film_modulation:
                    # Each modality gets a tiny condition generator from its
                    # post-stem mean/std statistics.  The final layer is
                    # zero-initialized so the extension starts as an exact
                    # identity map and cannot disrupt the pretrained path at
                    # initialization.
                    film_hidden = 8
                    self.modal_film = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(2 * len(group), film_hidden),
                                GELU(),
                                nn.Linear(film_hidden, 2 * len(group)),
                            )
                            for group in groups
                        ]
                    )
                    for generator in self.modal_film:
                        nn.init.zeros_(generator[-1].weight)
                        nn.init.zeros_(generator[-1].bias)
            self.modal_stem_act = GELU()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        long_stride = 25 if self.dense_long_branch else 50
        long_pool_stride = 4 if self.dense_long_branch else 2
        self.features2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=400, stride=long_stride, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=long_pool_stride, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        if self.branch_calibration:
            # Identity-initialized calibration for the two temporal scales.
            # The gain can learn to attenuate an unreliable long-scale branch;
            # the embedding gives the shared AFR block an explicit scale ID.
            self.branch_gain = nn.Parameter(
                torch.ones(2, 128, 1) if self.channelwise_branch_calibration else torch.ones(2)
            )
            self.branch_embedding = nn.Parameter(torch.zeros(2, 128, 1))
        else:
            self.branch_gain = None
            self.branch_embedding = None
        self.p2_residual_block = P2ResidualTemporalBlock(128, kernel_size=3) if self.p2_residual else None
        self.inplanes = 128
        if afr_type not in ("se", "eca", "cbam"):
            raise ValueError("Unsupported afr_type: {}".format(afr_type))
        afr_block = {"se": SEBasicBlock, "eca": ECAFeatureBlock,
                     "cbam": CBAMFeatureBlock}[afr_type]
        if self.post_p3_mode == "projection_tce":
            self.AFR = nn.Sequential(
                nn.Conv1d(128, afr_reduced_cnn_size, kernel_size=1, bias=False),
                nn.BatchNorm1d(afr_reduced_cnn_size),
            )
        elif self.post_p3_mode == "direct_pool":
            self.AFR = nn.Identity()
        elif self.post_p3_mode == "full":
            self.AFR = self._make_layer(afr_block, afr_reduced_cnn_size, 1)
        else:
            raise ValueError("Unsupported post_p3_mode: {}".format(self.post_p3_mode))
        if self.afr_bypass:
            self.afr_bypass_proj = nn.Sequential(
                nn.Conv1d(128, afr_reduced_cnn_size, kernel_size=1, bias=False),
                nn.BatchNorm1d(afr_reduced_cnn_size),
            )
            self.afr_bypass_raw_scale = nn.Parameter(torch.zeros(()))
        else:
            self.afr_bypass_proj = None
            self.afr_bypass_raw_scale = None

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pid_temporal_stem(x)
        if self.modal_stem:
            if self.modality_groups is None:
                x = self.modal_stem_act(self.modal_stem_bn(self.modal_stem_conv(x)))
            else:
                parts = [x[:, list(group), :] for group in self.modality_groups]
                transformed = [stem(part) for stem, part in zip(self.modal_stem_groups, parts)]
                if self.cross_modal_attention:
                    # Attend across modality tokens on a reduced time grid.
                    token_maps = [proj(part) for proj, part in zip(self.cross_modal_proj, transformed)]
                    token_maps = [
                        F.avg_pool1d(token_map, kernel_size=32, stride=32, ceil_mode=True)
                        for token_map in token_maps
                    ]
                    tokens = [token_map.transpose(1, 2) for token_map in token_maps]
                    token_stack = torch.stack(tokens, dim=2)  # (B, T, modalities, D)
                    bsz, steps, mods, dim = token_stack.shape
                    flat = token_stack.reshape(bsz * steps, mods, dim)
                    attended, _ = self.cross_modal_attn(flat, flat, flat, need_weights=False)
                    attended = self.cross_modal_norm(attended + flat).reshape(bsz, steps, mods, dim)
                    transformed = [
                        transformed[idx]
                        + F.interpolate(
                            self.cross_modal_out[idx](attended[:, :, idx, :].transpose(1, 2)),
                            size=transformed[idx].shape[-1], mode="linear", align_corners=False,
                        )
                        for idx in range(mods)
                    ]
                if self.film_modulation:
                    modulated = []
                    for part, generator in zip(transformed, self.modal_film):
                        mean = part.mean(dim=-1)
                        std = part.std(dim=-1, unbiased=False)
                        condition = generator(torch.cat((mean, std), dim=1))
                        gamma, beta = condition.chunk(2, dim=1)
                        # Keep the modulation small; zero initialization above
                        # makes this an identity mapping at step zero.
                        gamma = 1.0 + 0.1 * torch.tanh(gamma)
                        beta = 0.1 * torch.tanh(beta)
                        modulated.append(
                            part * gamma.unsqueeze(-1) + beta.unsqueeze(-1)
                        )
                    transformed = modulated
                x = torch.cat(transformed, dim=1)
        x1 = self.features1(x)
        x2 = self.features2(x)
        if self.p2_residual_block is not None:
            x2 = self.p2_residual_block(x2)
        if self.branch_calibration:
            if self.channelwise_branch_calibration:
                # Bounded channel-wise gain and bias; initialization is identity.
                g1 = 1.0 + 0.1 * torch.tanh(self.branch_gain[0] - 1.0)
                g2 = 1.0 + 0.1 * torch.tanh(self.branch_gain[1] - 1.0)
                x1 = g1 * x1 + 0.1 * torch.tanh(self.branch_embedding[0])
                x2 = g2 * x2 + 0.1 * torch.tanh(self.branch_embedding[1])
            else:
                x1 = self.branch_gain[0] * x1 + self.branch_embedding[0]
                x2 = self.branch_gain[1] * x2 + self.branch_embedding[1]
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        afr_input = x_concat
        x_concat = self.AFR(afr_input)
        if self.afr_bypass_proj is not None:
            bypass_scale = 0.1 * torch.tanh(self.afr_bypass_raw_scale)
            x_concat = x_concat + bypass_scale * self.afr_bypass_proj(afr_input)
        return x_concat

##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)


    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class ChannelQualityGate(nn.Module):
    """Learn per-epoch channel weights from simple signal-quality features."""

    def __init__(
        self,
        input_channels,
        channel_dropout=0.0,
        use_record_quality_prior=False,
        record_prior_min_scale=0.0,
    ):
        super(ChannelQualityGate, self).__init__()
        if input_channels < 2:
            raise ValueError("ChannelQualityGate requires at least two input channels")
        if channel_dropout < 0.0 or channel_dropout >= 1.0:
            raise ValueError(
                "channel_dropout must be in [0, 1), got {}".format(channel_dropout)
            )
        if record_prior_min_scale < 0.0:
            raise ValueError(
                "record_prior_min_scale must be non-negative, got {}".format(
                    record_prior_min_scale
                )
            )

        self.input_channels = input_channels
        self.channel_dropout = channel_dropout
        self.use_record_quality_prior = use_record_quality_prior
        self.record_prior_min_scale = float(record_prior_min_scale)
        hidden_size = max(8, input_channels * 4)
        self.score_network = nn.Sequential(
            nn.Linear(input_channels * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_channels),
        )
        self.channel_bias = nn.Parameter(torch.zeros(input_channels))
        self.record_prior_log_scale = (
            nn.Parameter(
                torch.tensor(
                    -3.0 if self.record_prior_min_scale > 0.0 else 0.5413248546
                )
            )
            if use_record_quality_prior
            else None
        )
        self.last_channel_weights = None

        # Start exactly at uniform early fusion. The gate then learns only when
        # validation-supported gradients identify useful channel reweighting.
        nn.init.zeros_(self.score_network[-1].weight)
        nn.init.zeros_(self.score_network[-1].bias)

    def _quality_features(self, x):
        # Inputs are stored in volts. Convert summary amplitudes to microvolts
        # before log compression so gradients remain numerically well scaled.
        rms_uv = torch.sqrt(torch.mean(x * x, dim=-1).clamp_min(1e-18)) * 1e6
        diff = x[..., 1:] - x[..., :-1]
        diff_rms_uv = torch.sqrt(torch.mean(diff * diff, dim=-1).clamp_min(1e-18)) * 1e6
        mean_abs_uv = torch.mean(torch.abs(x), dim=-1) * 1e6
        features = torch.stack((rms_uv, diff_rms_uv, mean_abs_uv), dim=-1)
        return torch.log1p(features).reshape(x.shape[0], -1)

    def forward(self, x, channel_mask=None, channel_prior=None):
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(
                "Expected input shape (batch, {}, samples), got {}".format(
                    self.input_channels, tuple(x.shape)
                )
            )

        logits = self.score_network(self._quality_features(x)) + self.channel_bias
        if self.use_record_quality_prior:
            if channel_prior is None:
                raise ValueError("record-quality-prior fusion requires channel_prior")
            channel_prior = torch.as_tensor(
                channel_prior,
                device=x.device,
                dtype=x.dtype,
            )
            if channel_prior.ndim == 1:
                channel_prior = channel_prior.unsqueeze(0).expand(x.shape[0], -1)
            if channel_prior.shape != logits.shape:
                raise ValueError(
                    "Expected channel_prior shape {} or ({},), got {}".format(
                        tuple(logits.shape),
                        self.input_channels,
                        tuple(channel_prior.shape),
                    )
                )
            prior_scale = self.record_prior_min_scale + F.softplus(
                self.record_prior_log_scale
            )
            logits = logits + prior_scale * torch.log(channel_prior.clamp_min(1e-4))
        elif channel_prior is not None:
            raise ValueError("channel_prior requires record-quality-prior fusion")
        keep = torch.ones_like(logits, dtype=torch.bool)
        if channel_mask is not None:
            channel_mask = torch.as_tensor(channel_mask, device=x.device, dtype=torch.bool)
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(x.shape[0], -1)
            if channel_mask.shape != logits.shape:
                raise ValueError(
                    "Expected channel_mask shape {} or ({},), got {}".format(
                        tuple(logits.shape), self.input_channels, tuple(channel_mask.shape)
                    )
                )
            keep = keep & channel_mask
        if self.training and self.channel_dropout > 0.0:
            keep = keep & (torch.rand_like(logits) >= self.channel_dropout)
        if channel_mask is not None or (self.training and self.channel_dropout > 0.0):
            all_dropped = ~keep.any(dim=1)
            if torch.any(all_dropped):
                fallback = torch.argmax(logits, dim=1)
                rows = torch.nonzero(all_dropped, as_tuple=False).squeeze(1)
                keep[rows, fallback[rows]] = True
            logits = logits.masked_fill(~keep, -1e4)

        weights = torch.softmax(logits, dim=1) * self.input_channels
        self.last_channel_weights = weights.detach()
        return x * weights.unsqueeze(-1)


class ChannelAttentionGate(nn.Module):
    """Cross-channel attention gate using compact per-channel signal features."""

    def __init__(
        self,
        input_channels,
        channel_dropout=0.0,
        feature_dim=32,
        num_heads=1,
    ):
        super(ChannelAttentionGate, self).__init__()
        if input_channels < 2:
            raise ValueError("ChannelAttentionGate requires at least two input channels")
        if channel_dropout < 0.0 or channel_dropout >= 1.0:
            raise ValueError(
                "channel_dropout must be in [0, 1), got {}".format(channel_dropout)
            )
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        self.input_channels = input_channels
        self.channel_dropout = channel_dropout
        self.feature_dim = feature_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.channel_embedding = nn.Parameter(torch.zeros(input_channels, feature_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.score = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 1),
        )
        self.channel_bias = nn.Parameter(torch.zeros(input_channels))
        self.last_channel_weights = None

        # Start close to uniform early fusion, then let validation-supported
        # gradients learn channel preference and cross-channel interaction.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _channel_features(self, x):
        rms_uv = torch.sqrt(torch.mean(x * x, dim=-1).clamp_min(1e-18)) * 1e6
        diff = x[..., 1:] - x[..., :-1]
        diff_rms_uv = torch.sqrt(torch.mean(diff * diff, dim=-1).clamp_min(1e-18)) * 1e6
        mean_abs_uv = torch.mean(torch.abs(x), dim=-1) * 1e6
        peak_uv = torch.amax(torch.abs(x), dim=-1) * 1e6
        features = torch.stack((rms_uv, diff_rms_uv, mean_abs_uv, peak_uv), dim=-1)
        return torch.log1p(features)

    def forward(self, x, channel_mask=None, channel_prior=None):
        if channel_prior is not None:
            raise ValueError("ChannelAttentionGate does not use channel_prior")
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(
                "Expected input shape (batch, {}, samples), got {}".format(
                    self.input_channels, tuple(x.shape)
                )
            )

        features = self.feature_proj(self._channel_features(x))
        features = features + self.channel_embedding.unsqueeze(0)
        attended, _ = self.attention(features, features, features, need_weights=False)
        logits = self.score(attended).squeeze(-1) + self.channel_bias

        keep = torch.ones_like(logits, dtype=torch.bool)
        if channel_mask is not None:
            channel_mask = torch.as_tensor(channel_mask, device=x.device, dtype=torch.bool)
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(x.shape[0], -1)
            if channel_mask.shape != logits.shape:
                raise ValueError(
                    "Expected channel_mask shape {} or ({},), got {}".format(
                        tuple(logits.shape), self.input_channels, tuple(channel_mask.shape)
                    )
                )
            keep = keep & channel_mask
        if self.training and self.channel_dropout > 0.0:
            keep = keep & (torch.rand_like(logits) >= self.channel_dropout)
        if channel_mask is not None or (self.training and self.channel_dropout > 0.0):
            all_dropped = ~keep.any(dim=1)
            if torch.any(all_dropped):
                fallback = torch.argmax(logits, dim=1)
                rows = torch.nonzero(all_dropped, as_tuple=False).squeeze(1)
                keep[rows, fallback[rows]] = True
            logits = logits.masked_fill(~keep, -1e4)

        weights = torch.softmax(logits, dim=1) * self.input_channels
        self.last_channel_weights = weights.detach()
        return x * weights.unsqueeze(-1)


class FeatureChannelAttentionFusion(nn.Module):
    """Fuse per-channel MRCNN feature maps with channel-level attention."""

    def __init__(
        self,
        input_channels,
        afr_reduced_cnn_size,
        feature_length,
        channel_dropout=0.0,
        use_record_quality_prior=True,
        record_prior_min_scale=1.0,
        token_dim=128,
        num_heads=4,
    ):
        super(FeatureChannelAttentionFusion, self).__init__()
        if input_channels < 2:
            raise ValueError(
                "FeatureChannelAttentionFusion requires at least two input channels"
            )
        if channel_dropout < 0.0 or channel_dropout >= 1.0:
            raise ValueError(
                "channel_dropout must be in [0, 1), got {}".format(channel_dropout)
            )
        embed_dim = afr_reduced_cnn_size * feature_length
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.input_channels = input_channels
        self.afr_reduced_cnn_size = afr_reduced_cnn_size
        self.feature_length = feature_length
        self.channel_dropout = channel_dropout
        self.use_record_quality_prior = use_record_quality_prior
        self.record_prior_min_scale = float(record_prior_min_scale)
        self.single_channel_mrcnn = MRCNN(afr_reduced_cnn_size, input_channels=1)
        self.token_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.channel_embedding = nn.Parameter(torch.zeros(input_channels, token_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.score = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
        )
        self.channel_bias = nn.Parameter(torch.zeros(input_channels))
        self.record_prior_log_scale = (
            nn.Parameter(torch.tensor(-3.0))
            if use_record_quality_prior
            else None
        )
        self.last_channel_weights = None

        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, x, channel_mask=None, channel_prior=None):
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(
                "Expected input shape (batch, {}, samples), got {}".format(
                    self.input_channels, tuple(x.shape)
                )
            )
        if self.use_record_quality_prior and channel_prior is None:
            raise ValueError("feature-attention-gated fusion requires channel_prior")
        if not self.use_record_quality_prior and channel_prior is not None:
            raise ValueError("channel_prior requires record-quality-prior fusion")

        batch_size, input_channels, samples = x.shape
        per_channel = x.reshape(batch_size * input_channels, 1, samples)
        feature_maps = self.single_channel_mrcnn(per_channel)
        if (
            feature_maps.shape[1] != self.afr_reduced_cnn_size
            or feature_maps.shape[2] != self.feature_length
        ):
            raise ValueError(
                "Unexpected MRCNN feature shape {}, expected (*, {}, {})".format(
                    tuple(feature_maps.shape),
                    self.afr_reduced_cnn_size,
                    self.feature_length,
                )
            )
        feature_maps = feature_maps.reshape(
            batch_size,
            input_channels,
            self.afr_reduced_cnn_size,
            self.feature_length,
        )
        tokens = self.token_proj(feature_maps.reshape(batch_size, input_channels, -1))
        tokens = tokens + self.channel_embedding.unsqueeze(0)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        logits = self.score(attended).squeeze(-1) + self.channel_bias

        if self.use_record_quality_prior:
            channel_prior = torch.as_tensor(
                channel_prior,
                device=x.device,
                dtype=x.dtype,
            )
            if channel_prior.ndim == 1:
                channel_prior = channel_prior.unsqueeze(0).expand(batch_size, -1)
            if channel_prior.shape != logits.shape:
                raise ValueError(
                    "Expected channel_prior shape {} or ({},), got {}".format(
                        tuple(logits.shape),
                        self.input_channels,
                        tuple(channel_prior.shape),
                    )
                )
            prior_scale = self.record_prior_min_scale + F.softplus(
                self.record_prior_log_scale
            )
            logits = logits + prior_scale * torch.log(channel_prior.clamp_min(1e-4))

        keep = torch.ones_like(logits, dtype=torch.bool)
        if channel_mask is not None:
            channel_mask = torch.as_tensor(channel_mask, device=x.device, dtype=torch.bool)
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1)
            if channel_mask.shape != logits.shape:
                raise ValueError(
                    "Expected channel_mask shape {} or ({},), got {}".format(
                        tuple(logits.shape), self.input_channels, tuple(channel_mask.shape)
                    )
                )
            keep = keep & channel_mask
        if self.training and self.channel_dropout > 0.0:
            keep = keep & (torch.rand_like(logits) >= self.channel_dropout)
        if channel_mask is not None or (self.training and self.channel_dropout > 0.0):
            all_dropped = ~keep.any(dim=1)
            if torch.any(all_dropped):
                fallback = torch.argmax(logits, dim=1)
                rows = torch.nonzero(all_dropped, as_tuple=False).squeeze(1)
                keep[rows, fallback[rows]] = True
            logits = logits.masked_fill(~keep, -1e4)

        weights = torch.softmax(logits, dim=1) * self.input_channels
        self.last_channel_weights = weights.detach()
        return torch.sum(feature_maps * weights[:, :, None, None], dim=1)



class ZeroInitializedLogitResidual(nn.Module):
    """Linear logit residual that does not perturb the model-init RNG stream."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_classes, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features):
        return F.linear(features, self.weight, self.bias)


class AttnSleep(nn.Module):
    def __init__(
        self,
        input_channels=1,
        channel_fusion="early",
        channel_dropout=0.0,
        record_prior_min_scale=0.0,
        afr_type="se",
        classifier_head="flatten",
        modal_stem=False,
        modality_groups=None,
        cross_modal_attention=False,
        film_modulation=False,
        branch_calibration=False,
        dense_long_branch=False,
        channelwise_branch_calibration=False,
        p2_residual=False,
        afr_bypass=False,
        post_p3_mode="full",
        pid_temporal_mode="none",
        pid_integral_kernel=101,
        pid_derivative_smooth_kernel=5,
        pid_refinement_kernel=9,
        pid_max_residual_scale=0.5,
        pid_integral_kernels=(51, 101, 201, 401),
        pid_derivative_smooth_kernels=(3, 5, 11, 21),
        pid_dynamic_channel_embedding_dim=4,
        pid_dynamic_gate_hidden_size=16,
        entropy_feature_dim=0,
    ):
        super(AttnSleep, self).__init__()

        N = 2  # number of TCE clones
        d_model = 80  # set to be 100 for SHHS dataset
        d_ff = 120   # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        if afr_type not in ("se", "eca", "cbam"):
            raise ValueError("Unsupported afr_type: {}".format(afr_type))
        self.afr_type = afr_type
        if classifier_head not in ("flatten", "attentive_pool"):
            raise ValueError("Unsupported classifier_head: {}".format(classifier_head))
        self.classifier_head = classifier_head
        if post_p3_mode not in ("full", "projection_tce", "direct_pool"):
            raise ValueError("Unsupported post_p3_mode: {}".format(post_p3_mode))
        self.post_p3_mode = post_p3_mode
        self.modal_stem = bool(modal_stem)
        self.entropy_feature_dim = int(entropy_feature_dim)
        if self.entropy_feature_dim < 0:
            raise ValueError("entropy_feature_dim must be non-negative")

        quality_fusions = (
            "quality-gated",
            "quality-prior-gated",
            "quality-hybrid-gated",
            "quality-piecewise-gated",
        )
        attention_fusions = ("attention-gated", "feature-attention-gated")
        if channel_fusion not in ("early",) + quality_fusions + attention_fusions:
            raise ValueError("Unsupported channel_fusion: {}".format(channel_fusion))
        if channel_fusion in quality_fusions + attention_fusions and input_channels < 2:
            raise ValueError("{} fusion requires at least two channels".format(channel_fusion))
        self.channel_fusion = channel_fusion
        if channel_fusion in quality_fusions:
            self.channel_gate = (
            ChannelQualityGate(
                input_channels,
                channel_dropout,
                use_record_quality_prior=channel_fusion
                in (
                    "quality-prior-gated",
                    "quality-hybrid-gated",
                    "quality-piecewise-gated",
                ),
                record_prior_min_scale=record_prior_min_scale,
            )
            )
        elif channel_fusion in attention_fusions:
            if channel_fusion == "attention-gated":
                self.channel_gate = ChannelAttentionGate(
                    input_channels,
                    channel_dropout=channel_dropout,
                )
            else:
                self.channel_gate = FeatureChannelAttentionFusion(
                    input_channels,
                    afr_reduced_cnn_size=afr_reduced_cnn_size,
                    feature_length=d_model,
                    channel_dropout=channel_dropout,
                    use_record_quality_prior=True,
                    record_prior_min_scale=record_prior_min_scale,
                )
        else:
            self.channel_gate = None
        self.mrcnn = (
            None
            if channel_fusion == "feature-attention-gated"
            else MRCNN(
                afr_reduced_cnn_size,
                input_channels=input_channels,
                afr_type=afr_type,
                modal_stem=modal_stem,
                modality_groups=modality_groups,
                cross_modal_attention=cross_modal_attention,
                film_modulation=film_modulation,
                branch_calibration=branch_calibration,
                dense_long_branch=dense_long_branch,
                channelwise_branch_calibration=channelwise_branch_calibration,
                p2_residual=p2_residual,
                afr_bypass=afr_bypass,
                post_p3_mode=post_p3_mode,
                pid_temporal_mode=pid_temporal_mode,
                pid_integral_kernel=pid_integral_kernel,
                pid_derivative_smooth_kernel=pid_derivative_smooth_kernel,
                pid_refinement_kernel=pid_refinement_kernel,
                pid_max_residual_scale=pid_max_residual_scale,
                pid_integral_kernels=pid_integral_kernels,
                pid_derivative_smooth_kernels=pid_derivative_smooth_kernels,
                pid_dynamic_channel_embedding_dim=pid_dynamic_channel_embedding_dim,
                pid_dynamic_gate_hidden_size=pid_dynamic_gate_hidden_size,
            )
        ) # use MRCNN_SHHS for SHHS dataset

        if self.post_p3_mode == "direct_pool":
            self.tce = None
        else:
            attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        if classifier_head == "attentive_pool":
            self.pool_score = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
            self.fc = nn.Linear(d_model, num_classes)
        elif self.post_p3_mode == "direct_pool":
            self.pool_score = None
            self.fc = nn.Linear(2 * 128, num_classes)
        else:
            self.pool_score = None
            self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)
        self.entropy_residual_head = (
            ZeroInitializedLogitResidual(self.entropy_feature_dim, num_classes)
            if self.entropy_feature_dim > 0
            else None
        )

    def forward(self, x, channel_mask=None, channel_prior=None, entropy_features=None):
        if self.channel_fusion == "feature-attention-gated":
            x_feat = self.channel_gate(
                x,
                channel_mask=channel_mask,
                channel_prior=channel_prior,
            )
        elif self.channel_gate is not None:
            x = self.channel_gate(
                x,
                channel_mask=channel_mask,
                channel_prior=channel_prior,
            )
            x_feat = self.mrcnn(x)
        elif channel_mask is not None or channel_prior is not None:
            raise ValueError(
                "channel_mask and channel_prior require a quality-aware fusion"
            )
        else:
            x_feat = self.mrcnn(x)
        if self.post_p3_mode == "direct_pool":
            # Shape-preserving P3 diagnostic head: temporal mean and std,
            # matching the frozen-probe aggregation without AFR/TCE.
            encoded_features = torch.cat(
                (x_feat.mean(dim=-1), x_feat.std(dim=-1, unbiased=False)), dim=1
            )
            final_output = self.fc(encoded_features)
            return self._add_entropy_residual(final_output, entropy_features)
        encoded_features = self.tce(x_feat)
        if self.classifier_head == "attentive_pool":
            # TCE output is (batch, tokens, d_model); use a learned convex
            # combination of tokens instead of flattening all token features.
            pool_logits = self.pool_score(encoded_features).squeeze(-1)
            pool_weights = torch.softmax(pool_logits, dim=1)
            encoded_features = torch.sum(
                encoded_features * pool_weights.unsqueeze(-1), dim=1
            )
        else:
            encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return self._add_entropy_residual(final_output, entropy_features)

    def _add_entropy_residual(self, logits, entropy_features):
        if self.entropy_residual_head is None:
            if entropy_features is not None:
                raise ValueError("entropy_features supplied without an entropy residual head")
            return logits
        if entropy_features is None:
            raise ValueError("entropy_features are required by the configured residual head")
        if entropy_features.ndim != 2 or entropy_features.shape != (
            logits.shape[0], self.entropy_feature_dim
        ):
            raise ValueError(
                "Expected entropy_features shape ({}, {}), got {}".format(
                    logits.shape[0], self.entropy_feature_dim, tuple(entropy_features.shape)
                )
            )
        return logits + self.entropy_residual_head(entropy_features)

######################################################################

class MRCNN_SHHS(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_SHHS, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat
