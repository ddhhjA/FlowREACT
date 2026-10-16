import torch
import torch.nn as nn
from torch import Tensor

from framework.motion_flow.Emotion3DMMTrans import Emotion3DMMTransformer
from framework.motion_transvae.BasicBlock import ConvBlock, PositionalEncoding, init_biased_mask
import torch.nn.functional as F
import math
from torchdiffeq import odeint
from hydra.utils import to_absolute_path
from framework.utils.model_mode import temporary_eval

# 定义当前版本，便于下面打印或者日志记录
__version__ = "FlowReact_v1.40_260117 V27 + pretrain_emotion + history_controls_20261002"
print(__version__)


def lengths_to_mask(lengths,
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.as_tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, _3dmm_dim=58, audio_dim=768, feature_dim=128, nhead=4, num_layers=4, device='cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.n_modality = 3  # expr, pose, audio

        # Projection (keep your lightweight design)
        self.expr_proj = nn.Linear(52, feature_dim)
        self.pose_proj = nn.Linear(6, feature_dim)
        self.audio_proj = nn.Linear(audio_dim, feature_dim)

        # 🔑 Modality embedding: (3, F) —— learnable modality ID
        self.modality_embed = nn.Parameter(torch.randn(self.n_modality, feature_dim) * 0.02)

        # 🔑 Time-aware positional encoding (up to max_len=1000)
        self.pos_encoding = PositionalEncoding(feature_dim, dropout=0.0, max_len=1000)
        assert self.pos_encoding.pe.shape[2] == self.feature_dim, \
            f"PE dim {self.pos_encoding.pe.shape[2]} != feature_dim {self.feature_dim}"

        # 🔑 Modality-aware attention mask: within same time step, allow full cross-modality;
        # across time steps, only allow causal or full (configurable)
        # We'll use full attention here (offline), can switch to causal for online
        self.causal = False  # set True for strict online

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, dim_feedforward=feature_dim * 2,
            batch_first=True, dropout=0.1
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.modality_gate = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def _create_modality_time_mask(self, T: int, device: torch.device):
        """
        Returns mask for (T*3, T*3) where:
        - Within same time step t: all 3 modalities attend to each other (no mask)
        - Across time: full attention (or causal if self.causal=True)
        """
        # Create time-step index for each token: [0,0,0, 1,1,1, ..., T-1,T-1,T-1]
        time_ids = torch.arange(T, device=device).repeat_interleave(3)  # (T*3,)
        # Create modality index: [0,1,2, 0,1,2, ...]
        mod_ids = torch.tensor([0,1,2] * T, device=device)  # (T*3,)

        # Full attention mask: True = ignore (PyTorch convention)
        mask = torch.zeros(T*3, T*3, dtype=torch.bool, device=device)

        if self.causal:
            # Only allow attending to past or current time steps
            time_diff = time_ids.unsqueeze(1) - time_ids.unsqueeze(0)  # (T*3, T*3)
            mask = time_diff < 0  # future time → mask

        # No need to mask within same time step — we WANT cross-modality fusion at same t!
        # So mask only across time (already handled above)

        return mask

    def forward(self, _3dmm, audio):
        """
        _3dmm: (B, T, 58)
        audio: (B, T, 768)
        Returns: (B, T, feature_dim)
        """
        B, T, _ = _3dmm.shape

        # Project each modality
        expr = self.expr_proj(_3dmm[..., :52])   # (B, T, F)
        pose = self.pose_proj(_3dmm[..., 52:])  # (B, T, F)
        audio_feat = self.audio_proj(audio)     # (B, T, F)

        # Stack modalities: (B, T, 3, F)
        fused = torch.stack([expr, pose, audio_feat], dim=2)  # (B, T, 3, F)

        # Add modality embedding
        fused = fused + self.modality_embed.unsqueeze(0).unsqueeze(0)  # (B, T, 3, F)

        # Reshape to (B, T*3, F) for Transformer
        fused = fused.view(B, T * 3, self.feature_dim)  # (B, T*3, F)

        # Add time-aware positional encoding (repeated for each modality)
        # pos_enc: (1, T, F) → repeat modality dim → (1, T*3, F)
        pos_enc = self.pos_encoding.pe[:T, 0, :]  # (T, D) ← squeeze out the 1-dim
        pos_enc = pos_enc.unsqueeze(0)  # (1, T, D)
        pos_enc = pos_enc.repeat(1, 1, 3).view(1, T * 3, self.feature_dim)  # (1, T*3, D)
        fused = fused + pos_enc

        # Create attention mask
        mask = self._create_modality_time_mask(T, device=_3dmm.device)  # (T*3, T*3)

        # Transformer encoding
        encoded = self.temporal_encoder(fused, mask=mask)  # (B, T*3, F)

        # 🔑 Critical: Instead of mean, take ONLY the 'expr' token per time step
        # Why expr? It's richest facial signal; or take learned weighted sum:
        encoded = encoded.view(B, T, 3, self.feature_dim)  # (B, T, 3, F)
        # Option 1: fixed — use expr token (most stable)
        gate_logits = self.modality_gate(encoded).squeeze(-1)  # (B, T, 3)
        gate = torch.softmax(gate_logits, dim=2).unsqueeze(-1)  # (B, T, 3, 1)
        out = (encoded * gate).sum(dim=2)  # (B, T, F)

        return self.out_proj(out)


class FrameAutoEncoderV2(nn.Module):
    """
    Frame-level autoencoder using Transformer instead of plain MLP.
    """

    def __init__(self, in_dim=58, z_dim=64, hidden=128, nhead=4, num_layers=2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_z = nn.Linear(hidden, z_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.from_z = nn.Linear(z_dim, hidden)
        self.out_proj = nn.Linear(hidden, in_dim)

    def encode(self, x):
        # x: (B,T,D)
        h = self.in_proj(x)  # (B,T,H)
        h = self.encoder(h)  # Transformer sequence modeling
        z = self.to_z(h)  # (B,T,Z)
        return z

    def decode(self, z):
        h = self.from_z(z)  # (B,T,H)
        h = self.decoder(h)
        x_recon = self.out_proj(h)  # (B,T,D)
        return x_recon


def sinusoidal_time_embedding(t, dim=32):
    """
    t: (B*T, 1) in [0,1]
    returns: (B*T, dim)
    """
    half_dim = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


class VelocityField(nn.Module):
    """
    Transformer-based velocity predictor.
    """

    def __init__(self, z_dim=64, cond_dim=64, hidden=128, nhead=4, num_layers=2):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.in_proj = nn.Linear(z_dim + hidden + cond_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(hidden, z_dim)

    def forward(self, zt, t, cond):
        B, T, _ = zt.shape
        t_emb = sinusoidal_time_embedding(t.view(B * T, 1), dim=128)
        t_feat = self.t_mlp(t_emb).view(B, T, -1)  # (B,T,H)

        inp = torch.cat([zt, t_feat, cond], dim=-1)  # (B,T,D)
        h = self.in_proj(inp)  # (B,T,H)
        h = self.transformer(h)  # sequence modeling
        v = self.out_proj(h)  # (B,T,Z)
        return v


# ---------------------------
# Flow Matching Decoder (真实 FM)
# ---------------------------
class FlowMatchingDecoderFM(nn.Module):
    def __init__(self,
                 output_3dmm_dim=58,
                 feature_dim=128,
                 z_dim=64,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.output_3dmm_dim = output_3dmm_dim
        self.cond_dim = feature_dim
        self.z_dim = z_dim
        self.window_size = 8
        # 只保留编码器用来把 GT 3DMM -> z0（teacher latent）
        # self.frame_ae = FrameAutoEncoder(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)
        # self.frame_ae = HierarchicalFrameAE(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)
        # self.frame_ae = ARKitHierarchicalFrameAE(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)
        # self.frame_ae = ARKitTwoStreamFrameAE(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)
        # self.frame_ae = DynamicFrequencyFrameAE(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)
        self.frame_ae = FrameAutoEncoderV2(in_dim=output_3dmm_dim, z_dim=z_dim, hidden=256)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=z_dim,
            kdim=self.cond_dim,
            vdim=self.cond_dim,
            num_heads=4,
            batch_first=True
        )
        self.v_net = VelocityField(z_dim=z_dim, cond_dim=z_dim, hidden=512)
        # self.emotion_fc = nn.Linear(self.cond_dim, 25)
        self.cond_proj = nn.Linear(self.cond_dim, self.z_dim)  # 备用

    def encode_frames_to_latent(self, frames):
        # frames: (B,T,58)
        return self.frame_ae.encode(frames)  # (B,T,z)

    def forward(self, encoded_feature, speaker_3dmm=None, listener_gt=None):
        assert listener_gt is not None, "listener_gt required for flow-matching training forward"
        B, T, _ = listener_gt.shape
        cond = encoded_feature  # (B,T,cond_dim)

        # teacher latent from GT
        z0 = self.encode_frames_to_latent(listener_gt)  # (B,T,z)

        # prior z1
        z1 = torch.randn_like(z0, device=z0.device)

        # sample t in [0,1]
        t = torch.rand(B, T, device=z0.device)

        # z_t on linear path
        zt = (1.0 - t.unsqueeze(-1)) * z0 + t.unsqueeze(-1) * z1
        v_true = (z1 - z0)

        # cond cross-attn
        zq, _ = self.cross_attn(zt, cond, cond)  # (B,T,z)

        # predict velocity
        v_pred = self.v_net(zt, t, zq)  # (B,T,z)

        outputs = {
            'z0': z0,
            'z1': z1,
            'zt': zt,
            'v_pred': v_pred,
            'v_true': v_true,
        }
        return outputs

    @torch.no_grad()
    def sample_latent_from_flow(self, encoded_feature, num_steps=2, stochastic=True, solver='dpm'):
        with temporary_eval(self):
            return self._sample_latent_from_flow(encoded_feature, num_steps, stochastic, solver)

    def sample_latent_from_flow_with_grad(self, encoded_feature, num_steps=2, stochastic=True, solver='dpm'):
        """Use the inference sampler with autograd for auxiliary training losses."""
        if not torch.is_grad_enabled():
            raise RuntimeError("Differentiable sampling requires enabled autograd")
        with temporary_eval(self):
            return self._sample_latent_from_flow(encoded_feature, num_steps, stochastic, solver)

    def _sample_latent_from_flow(self, encoded_feature, num_steps, stochastic, solver):
        # 返回 z 序列，不再解码为 3DMM
        cond = encoded_feature  # (B,T,cond_dim)
        B, T, _ = cond.shape

        # # init latent at t=1
        # z_flat = torch.randn(B * T, self.z_dim, device=cond.device) if stochastic else \
        #     torch.zeros(B * T, self.z_dim, device=cond.device)

        # fvcore traces this path to estimate FLOPs, but JIT cannot serialize a
        # torch.Generator argument. The initial values do not affect operation
        # counts, so use a deterministic tensor only while tracing.
        if stochastic and torch.jit.is_tracing():
            z_flat = torch.zeros(B * T, self.z_dim, device=cond.device)
        elif stochastic:
            tmp_seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
            g = torch.Generator(device=cond.device).manual_seed(tmp_seed)
            z_flat = torch.randn(B * T, self.z_dim, device=cond.device, generator=g)
        else:
            z_flat = torch.zeros(B * T, self.z_dim, device=cond.device)

        def eval_v(z_flat_in, t_scalar):
            zt_resh = z_flat_in.view(B, T, self.z_dim)
            t_tensor = torch.full((B * T,), float(t_scalar), device=cond.device)
            t_resh = t_tensor.view(B, T)
            zq, _ = self.cross_attn(zt_resh, cond, cond)
            v = self.v_net(zt_resh, t_resh, zq)  # (B,T,z)
            return v.view(B * T, self.z_dim)

        ts = torch.linspace(1.0, 0.0, steps=max(1, num_steps), device=cond.device)

        def euler(z):
            for i in range(len(ts) - 1):
                t_n, t_np1 = float(ts[i]), float(ts[i + 1])
                h = t_np1 - t_n
                v_n = eval_v(z, t_n)
                z = z + h * v_n
            return z

        def heun(z):
            for i in range(len(ts) - 1):
                t_n, t_np1 = float(ts[i]), float(ts[i + 1])
                h = t_np1 - t_n
                v_n = eval_v(z, t_n)
                z_pred = z + h * v_n
                v_pred = eval_v(z_pred, t_np1)
                z = z + 0.5 * h * (v_n + v_pred)
            return z

        def rk4(z):
            for i in range(len(ts) - 1):
                t_n, t_np1 = float(ts[i]), float(ts[i + 1])
                h = t_np1 - t_n
                k1 = eval_v(z, t_n)
                k2 = eval_v(z + 0.5 * h * k1, t_n + 0.5 * h)
                k3 = eval_v(z + 0.5 * h * k2, t_n + 0.5 * h)
                k4 = eval_v(z + h * k3, t_np1)
                z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return z

        def dpm(z):
            N = len(ts) - 1
            if N <= 0:
                return z
            v_hist = []
            t0, t1 = float(ts[0]), float(ts[1])
            h = t1 - t0
            v0 = eval_v(z, t0)
            z_pred = z + h * v0
            v1 = eval_v(z_pred, t1)
            z = z + 0.5 * h * (v0 + v1)
            v_hist = [v1, v0]
            for i in range(1, N):
                t_n, t_np1 = float(ts[i]), float(ts[i + 1])
                h = t_np1 - t_n
                if i == 1:
                    v_n, v_nm1 = v_hist[0], v_hist[1]
                    z_next = z + h * (1.5 * v_n - 0.5 * v_nm1)
                    v_next = eval_v(z_next, t_np1)
                    v_hist = [v_next, v_n, v_nm1]
                    z = z_next
                else:
                    v_n, v_nm1, v_nm2 = v_hist[0], v_hist[1], v_hist[2]
                    z_next = z + h * ((23.0 / 12.0) * v_n + (-16.0 / 12.0) * v_nm1 + (5.0 / 12.0) * v_nm2)
                    v_next = eval_v(z_next, t_np1)
                    v_hist = [v_next, v_n, v_nm1]
                    z = z_next
            return z

        if solver == 'euler':
            z0_flat = euler(z_flat)
        elif solver == 'heun':
            z0_flat = heun(z_flat)
        elif solver == 'rk4':
            z0_flat = rk4(z_flat)
        elif solver == 'dpm':
            z0_flat = dpm(z_flat)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        z0 = z0_flat.view(B, T, self.z_dim)
        return z0

    def reset_window_size(self, window_size):
        self.window_size = window_size


class PeriodicFeatureExtractor(nn.Module):
    """
    从 encoded_feature 提取周期性特征 (B, T, cyc_dim)
    使用小型 Transformer 提取时序周期模式。
    """
    def __init__(self, feature_dim=128, cyc_dim=32, nhead=4, num_layers=2, hidden=128, dropout=0.1, max_len=1000):
        super().__init__()
        self.cyc_dim = cyc_dim
        self.feature_dim = feature_dim
        self.hidden = hidden

        # 投影到 Transformer hidden 维度
        self.in_proj = nn.Linear(feature_dim, hidden)

        # Positional encoding
        self.pos_emb = nn.Embedding(max_len, hidden)

        # 小型 Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden*2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出周期特征
        self.out_proj = nn.Linear(hidden, cyc_dim)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(cyc_dim)

    def forward(self, encoded_feature):
        """
        encoded_feature: (B, T, feature_dim)
        returns: cyc_feat (B, T, cyc_dim)
        """
        B, T, _ = encoded_feature.shape
        # 投影
        h = self.in_proj(encoded_feature)  # (B, T, hidden)

        # 添加位置编码
        pos_ids = torch.arange(T, device=encoded_feature.device)
        h = h + self.pos_emb(pos_ids).unsqueeze(0)  # (B, T, hidden)

        # Transformer 编码
        h = self.encoder(h)  # (B, T, hidden)

        # 投影到周期特征
        cyc_feat = self.out_proj(h)  # (B, T, cyc_dim)
        cyc_feat = self.norm(cyc_feat)

        return cyc_feat


def periodic_consistency_loss(
    periodic_extractor,
    encoded_feature,
    max_shift=8
):
    """
    周期一致性正则：
    P(x) ≈ shift_back(P(shift(x)))

    Args:
        periodic_extractor: PeriodicFeatureExtractor
        encoded_feature: (B, T, F) 训练时 T=256
        max_shift: 最大时间平移
    Returns:
        scalar loss
    """
    B, T, _ = encoded_feature.shape
    device = encoded_feature.device

    # 随机采样 shift
    tau = torch.randint(2, max_shift + 1, (1,), device=device).item()

    # 原始周期特征
    cyc = periodic_extractor(encoded_feature)  # (B,T,C)

    # 时间平移
    shifted_feat = torch.roll(encoded_feature, shifts=tau, dims=1)
    cyc_shifted = periodic_extractor(shifted_feat)

    # 对齐回来
    cyc_shifted = torch.roll(cyc_shifted, shifts=-tau, dims=1)

    # L2 consistency
    loss = F.mse_loss(cyc, cyc_shifted)

    return loss


class ExprHeadAttention(nn.Module):
    """
    Gated Expression → Head modulation.

    - Local expr→head cross-attention (main signal)
    - Mid / Global expr produce gating coefficients
    - Expr acts as modulation, NOT motion driver
    """
    def __init__(
        self,
        dim_expr,
        dim_head,
        nhead=8,
        dropout=0.1,
        mid_kernel=4
    ):
        super().__init__()

        self.dim_head = dim_head

        # ---- projections ----
        self.q_proj = nn.Linear(dim_head, dim_head)
        self.k_proj = nn.Linear(dim_expr, dim_head)
        self.v_proj = nn.Linear(dim_expr, dim_head)

        # ---- Local cross-attention (ONLY true attention) ----
        self.local_attn = nn.MultiheadAttention(
            embed_dim=dim_head,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # ---- Mid / Global pooling ----
        self.mid_pool = nn.AvgPool1d(
            kernel_size=mid_kernel,
            stride=mid_kernel
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Gating MLPs ----
        self.mid_gate = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.Sigmoid()
        )
        self.global_gate = nn.Sequential(
            nn.Linear(dim_head, dim_head),
            nn.Sigmoid()
        )

        # ---- FFN ----
        self.ffn = nn.Sequential(
            nn.Linear(dim_head, dim_head * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_head * 4, dim_head)
        )

        # ---- Norm ----
        self.norm_attn = nn.LayerNorm(dim_head)
        self.norm_ffn = nn.LayerNorm(dim_head)

    def forward(self, head_feat, expr_feat):
        """
        head_feat: (B, T, Dh)
        expr_feat: (B, T, De)
        """

        # ---- projections ----
        q = self.q_proj(head_feat)
        k = self.k_proj(expr_feat)
        v = self.v_proj(expr_feat)

        # =====================================================
        # 1) Local expr → head attention (core signal)
        # =====================================================
        local_out, _ = self.local_attn(q, k, v)
        head_feat = self.norm_attn(head_feat + local_out)

        # =====================================================
        # 2) Mid / Global gates (NO direct residual)
        # =====================================================
        # mid gate
        v_mid = self.mid_pool(v.transpose(1, 2)).transpose(1, 2)
        mid_summary = v_mid.mean(dim=1)          # (B, Dh)
        mid_gate = self.mid_gate(mid_summary)    # (B, Dh)

        # global gate
        v_global = self.global_pool(v.transpose(1, 2)).squeeze(-1)
        global_gate = self.global_gate(v_global) # (B, Dh)

        # combine gates
        gate = (mid_gate + global_gate) / 2.0
        gate = gate.unsqueeze(1)                 # (B,1,Dh)

        # modulation (NOT addition)
        head_feat = head_feat * (1.0 + 0.3 * gate)

        # =====================================================
        # 3) FFN
        # =====================================================
        ffn_out = self.ffn(head_feat)
        head_feat = self.norm_ffn(head_feat + ffn_out)

        return head_feat



class MotionTransformerV6(nn.Module):
    """
    Transformer-based motion generator with expression-only periodic modulation.
    """
    def __init__(self, z_dim=256, cond_dim=128, cyc_dim=64, out_dim=58,
                 hidden=512, nhead=8, num_layers=4, max_len=2000, dropout=0.1):
        super().__init__()

        assert hidden % 2 == 0
        self.hidden = hidden
        self.hidden_expr = hidden // 2
        self.hidden_head = hidden // 2

        # ---- projections ----
        self.z_proj = nn.Linear(z_dim, hidden)

        # cond 分成两个子空间
        self.c_proj_expr = nn.Linear(cond_dim, self.hidden_expr)
        self.c_proj_head = nn.Linear(cond_dim, self.hidden_head)

        # FiLM 只给 expression
        self.cyc_film = nn.Linear(cyc_dim, 2 * self.hidden_expr)

        # Expr → Head cross-attention
        self.expr_head_attn = ExprHeadAttention(
            dim_expr=self.hidden_expr,
            dim_head=self.hidden_head,
            nhead=8,
            dropout=dropout
        )

        self.type_emb = nn.Embedding(2, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, z_seq, cond_seq, cyc_feat=None):
        """
        z_seq:   (B, T, z_dim)
        cond_seq:(B, T, cond_dim)
        cyc_feat:(B, T, cyc_dim)
        """
        B, T, _ = z_seq.shape

        # ---- z branch ----
        z_h = self.z_proj(z_seq)
        z_h = z_h + self.type_emb(
            torch.zeros(T, device=z_seq.device, dtype=torch.long)
        ).unsqueeze(0)

        # ---- cond branch (split) ----
        c_expr = self.c_proj_expr(cond_seq)
        c_head = self.c_proj_head(cond_seq)

        # 🔑 周期只作用在 expression
        if cyc_feat is not None:
            gamma, beta = self.cyc_film(cyc_feat).chunk(2, dim=-1)
            c_expr = c_expr * (1.0 + gamma) + beta

        # 🔑 插这里
        c_head = self.expr_head_attn(c_head, c_expr)

        # 拼回 cond token
        c_h = torch.cat([c_expr, c_head], dim=-1)

        c_h = c_h + self.type_emb(
            torch.ones(T, device=z_seq.device, dtype=torch.long)
        ).unsqueeze(0)

        # ---- concat tokens ----
        tokens = torch.cat([z_h, c_h], dim=1)  # (B, 2T, H)

        pos_ids = torch.arange(tokens.size(1), device=tokens.device)
        tokens = tokens + self.pos_emb(pos_ids).unsqueeze(0)

        # ---- transformer ----
        h = self.encoder(tokens)
        h = self.norm(h)

        h_z = h[:, :T, :]
        out = self.out_proj(h_z)

        return out



class FlowReact(nn.Module):
    def __init__(self,
                 # keep same args you used
                 img_size=224, audio_dim=768,
                 output_3dmm_dim=58, output_emotion_dim=25,
                 feature_dim=128, seq_len=750,
                 task='online', window_size=8, device='cuda',
                 z_dim=256,
                 emotion_transfer_ckpt=None,
                 **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = True if task == 'online' else False
        self.window_size = window_size
        self.device = device
        self.cyc_dim = feature_dim // 2  # 周期特征维度

        # === keep your original SpeakerBehaviourEncoder ===
        # Assumes SpeakerBehaviourEncoder class exists in scope
        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(
            _3dmm_dim=output_3dmm_dim,
            audio_dim=audio_dim,
            feature_dim=feature_dim,
            device=device
        )

        # === Flow Matching Decoder (真实实现) ===
        # cond_dim == feature_dim (fusion output)
        self.reaction_decoder = FlowMatchingDecoderFM(
            output_3dmm_dim=output_3dmm_dim,
            feature_dim=feature_dim,
            z_dim=z_dim,
            device=device
        )

        # === Fusion (speaker feature + listener history) ===
        # keep your original fusion module (same shapes)
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim + output_emotion_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # === Motion Transformer with periodic feature injection ===
        self.periodic_extractor = PeriodicFeatureExtractor(
            feature_dim=feature_dim,
            cyc_dim=self.cyc_dim,
            nhead=4,
            num_layers=2,
            hidden=feature_dim,
            dropout=0.1,
            max_len=1000
        )

        self.motion_transformer = MotionTransformerV6(
            z_dim=z_dim, cond_dim=feature_dim, cyc_dim=self.cyc_dim,
            out_dim=output_3dmm_dim,
            hidden=feature_dim, nhead=8, num_layers=4,
            max_len=2000, dropout=0.1
        )

        self.emotion_transfer = Emotion3DMMTransformer(
            emotion_dim=output_emotion_dim,
            param_dim=output_3dmm_dim,
            latent_dim=256,
            n_layers=4,
            n_heads=4,
            window_size=16,
        )
        self._freeze_emotion_transfer = emotion_transfer_ckpt is not None

        # 加载预训练的 emotion_transfer 模型权重
        if emotion_transfer_ckpt is not None:
            ckpt_path = to_absolute_path(emotion_transfer_ckpt)
            state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
            self.emotion_transfer.load_state_dict(state_dict)
            # 冻结
            for param in self.emotion_transfer.parameters():
                param.requires_grad = False
            self.emotion_transfer.eval()
            self.emotion_transfer.to(self.device)
            self.emotion_transfer.reset_window_size(self.window_size)
            self.emotion_transfer.set_inference()
            print(f"Successfully loaded pre-trained Emotion3DMMTransformer from {ckpt_path}")




    def train(self, mode=True):
        super().train(mode)
        # Frozen converter dropout and random training windows must stay disabled.
        if self._freeze_emotion_transfer:
            self.emotion_transfer.eval()
            self.emotion_transfer.set_inference()
        return self

    # forward keeps same interface: will call reaction_decoder appropriately
    def forward(self, speaker_3dmm=None, speaker_audio=None, listener_gt=None, **kwargs):
        """
        - if listener_gt provided: training-mode -> run FM training forward (returns losses)
        - otherwise: sampling / inference -> return predicted sequences
        """
        history_mode = str(kwargs.get('history_mode', 'previous_window')).lower()
        if history_mode not in {'previous_window', 'legacy_zero'}:
            raise ValueError(f"Unknown listener history mode: {history_mode}")
        history_weight = float(kwargs.get('history_weight', 1.0))
        history_probability = float(kwargs.get('sampled_history_probability', 0.0))
        if not 0.0 <= history_weight <= 1.0:
            raise ValueError("history_weight must be in [0, 1]")
        if not 0.0 <= history_probability <= 1.0:
            raise ValueError("sampled_history_probability must be in [0, 1]")
        sampled_history = kwargs.get('sampled_history')
        if history_probability > 0 and (listener_gt is None or sampled_history is None or not self.online):
            raise ValueError("Sampled history requires online supervised forward and a generated rollout")
        if not self.online and history_weight != 1.0:
            raise ValueError("history_weight is only supported for online windows")
        ablation_mode = str(kwargs.get('ablation_mode', 'clean')).lower()
        if bool(kwargs.get('force_all_zero', False)):
            ablation_mode = 'hard_zero'
        force_all_zero = ablation_mode == 'hard_zero'
        zero_speaker_feature = ablation_mode in {
            'speaker_feature_zero',
            'speaker_condition_zero',
            'hard_zero',
        }
        zero_listener_history = ablation_mode in {'history_zero', 'hard_zero'}
        zero_flow_noise = ablation_mode == 'flow_noise_zero'
        zero_flow_latent = ablation_mode in {'flow_latent_zero', 'hard_zero'}
        zero_periodic_feature = ablation_mode in {'periodic_zero', 'hard_zero'}
        model_ablation_enabled = ablation_mode in {
            'speaker_feature_zero',
            'history_zero',
            'flow_noise_zero',
            'flow_latent_zero',
            'periodic_zero',
            'speaker_condition_zero',
            'hard_zero',
        }
        sample_with_grad = bool(kwargs.get('sample_with_grad', False))
        detach_generated_history = bool(kwargs.get('detach_generated_history', False))
        if sample_with_grad and (listener_gt is not None or not self.online or model_ablation_enabled):
            raise ValueError("Differentiable sampling requires an online, unablated generation forward")
        if sample_with_grad and not torch.is_grad_enabled():
            raise RuntimeError("Differentiable sampling requires enabled autograd")
        if detach_generated_history and not sample_with_grad:
            raise ValueError("Detached generated history is only supported for differentiable sampling")
        ablation_audit = None
        return_debug_tensors = bool(kwargs.get('return_debug_tensors', False))
        debug_window_index = max(0, int(kwargs.get('debug_window_index', 1)))
        debug_tensors = None

        # distribution = []
        fm_outputs = []

        if self.online:
            # === Online window-based (generation or training) ===
            # speaker_3dmm and speaker_audio expected stacked as in your original code
            speaker_3dmm = torch.stack(speaker_3dmm, dim=0).to(self.device)
            speaker_audio = torch.stack(speaker_audio, dim=0).to(self.device)

            # print("speaker_3dmm:", speaker_3dmm.shape)
            # print("speaker_audio:", speaker_audio.shape)

            motion_lengths = (
                torch.as_tensor(
                    kwargs.get('motion_lengths', [self.seq_len] * len(speaker_3dmm)),
                    device=self.device,
                ).clamp(max=self.seq_len)
            )

            frame_num = speaker_3dmm.shape[1]
            period = frame_num // self.window_size

            reaction_3dmm = torch.zeros((speaker_3dmm.size(0), self.window_size, self.output_3dmm_dim),
                                        device=self.device)
            reaction_emotion = torch.zeros((speaker_3dmm.size(0), self.window_size, self.output_emotion_dim),
                                           device=self.device)

            if listener_gt is not None:
                listener_gt = torch.stack(listener_gt, dim=0).to(self.device)

            if history_probability > 0:
                history_params = sampled_history['listener_3dmm_out'].detach().to(speaker_3dmm)
                history_emotions = sampled_history['listener_emotion_out'].detach().to(speaker_3dmm)
                expected = (speaker_3dmm.size(0), frame_num)
                if (history_params.shape != (*expected, self.output_3dmm_dim)
                        or history_emotions.shape != (*expected, self.output_emotion_dim)):
                    raise ValueError("Sampled history must match the batch and full input sequence shape")

            for i in range(0, period):
                # 当前 window
                start_idx = i * self.window_size
                end_idx = (i + 1) * self.window_size
                speaker_3dmm_ = speaker_3dmm[:, start_idx:end_idx]
                speaker_audio_ = speaker_audio[:, start_idx:end_idx]

                # Hard-zero bypasses learned speaker projections; the speaker-feature
                # ablation zeros their output while preserving the rest of the model.
                if force_all_zero:
                    speaker_feature = torch.zeros(
                        speaker_3dmm_.shape[0],
                        self.window_size,
                        self.feature_dim,
                        device=self.device,
                        dtype=speaker_3dmm_.dtype,
                    )
                else:
                    speaker_feature = self.speaker_behaviour_encoder(speaker_3dmm_, speaker_audio_)
                    if zero_speaker_feature:
                        speaker_feature = torch.zeros_like(speaker_feature)

                encoded_feature = speaker_feature

                # print("encoded_feature:", encoded_feature.shape)
                # print("reaction_3dmm:", reaction_3dmm.shape)
                # print("reaction_emotion:", reaction_emotion.shape)

                # 保证 reaction buffer 与当前 window 同步长度
                reaction_3dmm_window = reaction_3dmm[:, -self.window_size:, :]
                reaction_emotion_window = reaction_emotion[:, -self.window_size:, :]
                if detach_generated_history:
                    # Truncate cross-window backpropagation, retaining the actual history values.
                    reaction_3dmm_window = reaction_3dmm_window.detach()
                    reaction_emotion_window = reaction_emotion_window.detach()

                if i > 0 and history_probability > 0 and history_weight > 0 and not zero_listener_history:
                    # Use only the preceding generated window, never current/future GT.
                    previous = slice(start_idx - self.window_size, start_idx)
                    use_sample = torch.ones((speaker_3dmm.size(0), 1, 1), device=self.device, dtype=torch.bool)
                    if history_probability < 1.0:
                        use_sample = torch.rand(use_sample.shape, device=self.device) < history_probability
                    reaction_3dmm_window = torch.where(use_sample, history_params[:, previous], reaction_3dmm_window)
                    reaction_emotion_window = torch.where(use_sample, history_emotions[:, previous], reaction_emotion_window)

                if zero_listener_history or history_weight == 0 or (listener_gt is None and history_mode == 'legacy_zero'):
                    reaction_3dmm_window = torch.zeros_like(reaction_3dmm_window)
                    reaction_emotion_window = torch.zeros_like(reaction_emotion_window)
                elif history_weight != 1.0:
                    reaction_3dmm_window = history_weight * reaction_3dmm_window
                    reaction_emotion_window = history_weight * reaction_emotion_window

                # Hard-zero also bypasses the fusion layer and its learned bias.
                if not force_all_zero:
                    encoded_feature = self.fusion(
                        torch.cat((encoded_feature, reaction_3dmm_window, reaction_emotion_window), dim=-1)
                    )

                # If we are training and have listener_gt for this window, pass it to decoder for FM loss
                # listener_gt expected in kwargs as 'listener_gt' (full sequence) - we slice window here
                if listener_gt is not None:
                    gt_window = listener_gt[:, start_idx:end_idx]  # (B, window_size, 58)

                    # print("gt_window:", gt_window.shape)
                    # print("encoded_feature:", encoded_feature.shape)
                    fm_output = self.reaction_decoder.forward(
                        encoded_feature, speaker_3dmm=speaker_3dmm_, listener_gt=gt_window
                    )
                    fm_output['window_start'] = start_idx

                    # distribution.append(loss_dict)
                    # for online training we won't sample for generation pipeline; optionally sample for vis
                    z_teacher = fm_output['z0']  # (B,W,z)

                    # 提取周期特征
                    cyc_feat = self.periodic_extractor(encoded_feature)  # (B,W,cyc_dim)

                    loss_cyc = periodic_consistency_loss(
                        self.periodic_extractor,
                        encoded_feature,
                        max_shift=8
                    )

                    fm_output['loss_cyc'] = loss_cyc

                    # 2) 用 Transformer 生成当前 window 的 3DMM
                    pred_x = self.motion_transformer(z_teacher, encoded_feature, cyc_feat)  # (B,W,58)
                    pred_emotion = self.emotion_transfer.to_emotion(pred_x)[0]  # (B,W,25)

                    fm_output['recon'] = pred_x
                    fm_output['listener_emotion'] = pred_emotion
                    # update reaction buffers (take last window)
                    if i > 0:
                        reaction_3dmm = torch.cat(
                            (reaction_3dmm[:, :i * self.window_size], pred_x[:, -self.window_size:, :]), dim=1)
                        reaction_emotion = torch.cat(
                            (reaction_emotion[:, :i * self.window_size], pred_emotion[:, -self.window_size:, :]), dim=1)
                    else:
                        reaction_3dmm = pred_x[:, -self.window_size:, :]
                        reaction_emotion = pred_emotion[:, -self.window_size:, :]

                    fm_outputs.append(fm_output)

                else:
                    # inference / generation: sample
                    # 推理：先采样 latent，再用 Transformer 生成 3DMM
                    if zero_flow_latent:
                        z_sample = torch.zeros(
                            encoded_feature.shape[0],
                            encoded_feature.shape[1],
                            self.reaction_decoder.z_dim,
                            device=encoded_feature.device,
                            dtype=encoded_feature.dtype,
                        )
                    else:
                        sampler = (self.reaction_decoder.sample_latent_from_flow_with_grad
                                   if sample_with_grad else self.reaction_decoder.sample_latent_from_flow)
                        z_sample = sampler(
                            encoded_feature,
                            stochastic=not zero_flow_noise,
                        )

                    if zero_periodic_feature:
                        cyc_feat = torch.zeros(
                            encoded_feature.shape[0],
                            encoded_feature.shape[1],
                            self.cyc_dim,
                            device=encoded_feature.device,
                            dtype=encoded_feature.dtype,
                        )
                    else:
                        cyc_feat = self.periodic_extractor(encoded_feature)  # (B,W,cyc_dim)

                    if model_ablation_enabled:
                        ablation_audit = {
                            'mode': ablation_mode,
                            'speaker_feature_max_abs': float(speaker_feature.abs().max().item()),
                            'encoded_feature_max_abs': float(encoded_feature.abs().max().item()),
                            'flow_latent_max_abs': float(z_sample.abs().max().item()),
                            'periodic_feature_max_abs': float(cyc_feat.abs().max().item()),
                            'listener_history_used': not zero_listener_history and history_mode != 'legacy_zero' and history_weight > 0,
                            'flow_sampling_used': not zero_flow_latent,
                            'flow_initial_latent_zero': zero_flow_noise or zero_flow_latent,
                            'flow_stochastic': not zero_flow_noise and not zero_flow_latent,
                        }
                    pred_x = self.motion_transformer(z_sample, encoded_feature, cyc_feat)  # (B,W,58)
                    pred_emotion = self.emotion_transfer.to_emotion(pred_x)[0]  # (B,W,25)
                    if return_debug_tensors and debug_tensors is None and i == debug_window_index:
                        debug_tensors = {
                            'window_index': i,
                            'speaker_audio': speaker_audio_[:1].detach().cpu(),
                            'speaker_3dmm': speaker_3dmm_[:1].detach().cpu(),
                            'speaker_feature': speaker_feature[:1].detach().cpu(),
                            'listener_history_3dmm': reaction_3dmm_window[:1].detach().cpu(),
                            'listener_history_emotion': reaction_emotion_window[:1].detach().cpu(),
                            'encoded_feature': encoded_feature[:1].detach().cpu(),
                            'flow_latent': z_sample[:1].detach().cpu(),
                            'periodic_feature': cyc_feat[:1].detach().cpu(),
                            'prediction_3dmm': pred_x[:1].detach().cpu(),
                            'prediction_emotion': pred_emotion[:1].detach().cpu(),
                        }
                    # print("pred_x:", pred_x.shape)
                    # Keep only generated frames so the buffer tail is the latest reaction.
                    if i != 0:
                        reaction_3dmm = torch.cat((reaction_3dmm, pred_x), dim=1)
                        reaction_emotion = torch.cat((reaction_emotion, pred_emotion), dim=1)
                    else:
                        reaction_3dmm = pred_x
                        reaction_emotion = pred_emotion

                    # distribution.append({'sampled': True})

            # finalize outputs (truncate to actual frame_num)
            listener_3dmm_out = reaction_3dmm[:, :frame_num]
            # print("listener_3dmm_out:", listener_3dmm_out.shape)
            listener_emotion_out = reaction_emotion[:, :frame_num]

            # mask padding (same as before)
            seq_mask = lengths_to_mask(
                lengths=motion_lengths,
                device=speaker_3dmm.device,
                max_len=frame_num
            ).unsqueeze(-1).float()
            listener_3dmm_out = listener_3dmm_out * seq_mask
            listener_emotion_out = listener_emotion_out * seq_mask

            # print("listener_3dmm_out:", listener_3dmm_out.shape)
            # print("listener_emotion_out:", listener_emotion_out.shape)

            final_output = {
                'listener_3dmm_out': listener_3dmm_out,
                'listener_emotion_out': listener_emotion_out,
                'fm_outputs': fm_outputs,  # only for online training mode
                'history_mode': history_mode,
                'history_weight': history_weight,
            }
            if ablation_audit is not None:
                final_output['ablation_audit'] = ablation_audit
            if debug_tensors is not None:
                final_output['debug_tensors'] = debug_tensors

            # return in same format as before
            return final_output

        else:
            # === Offline: process each full sequence ===
            listener_3dmm_outs, listener_emotion_outs = [], []
            for speaker_3dmm_, speaker_audio_ in zip(speaker_3dmm, speaker_audio):
                speaker_3dmm_ = speaker_3dmm_.unsqueeze(0).to(self.device)
                speaker_audio_ = speaker_audio_.unsqueeze(0).to(self.device)

                encoded_feature = self.speaker_behaviour_encoder(speaker_3dmm_, speaker_audio_)
                encoded_feature = self.fusion(torch.cat((encoded_feature,
                                                         torch.zeros((1, encoded_feature.size(1), self.output_3dmm_dim),
                                                                     device=self.device),
                                                         torch.zeros(
                                                             (1, encoded_feature.size(1), self.output_emotion_dim),
                                                             device=self.device)),
                                                        dim=-1))

                # If listener_gt provided -> training forward
                if listener_gt is not None:
                    listener_gt_seq = listener_gt.unsqueeze(0).to(self.device)  # (1,T,58)
                    fm_output = self.reaction_decoder.forward(
                        encoded_feature, speaker_3dmm=speaker_3dmm_, listener_gt=listener_gt_seq
                    )
                    fm_outputs.append(fm_output)
                    # distribution.append(loss_dict)
                    # also produce sample for monitoring
                    pred_x, pred_emotion = self.reaction_decoder.sample_latent_from_flow(encoded_feature)
                    fm_output['recon'] = pred_x
                    fm_output['listener_emotion'] = pred_emotion
                    listener_3dmm_outs.append(pred_x.squeeze(0))
                    listener_emotion_outs.append(pred_emotion.squeeze(0))
                else:
                    pred_x, pred_emotion = self.reaction_decoder.sample_latent_from_flow(encoded_feature)
                    listener_3dmm_outs.append(pred_x.squeeze(0))
                    listener_emotion_outs.append(pred_emotion.squeeze(0))
                    # distribution.append({'sampled': True})

            # dict -> tensor
            listener_3dmm_outs = torch.stack(listener_3dmm_outs, dim=0)
            listener_emotion_outs = torch.stack(listener_emotion_outs, dim=0)

            final_output = {
                'listener_3dmm_outs': listener_3dmm_outs,
                'listener_emotion_outs': listener_emotion_outs,
                'fm_outputs': fm_outputs,  # only for offline training mode
            }

            return final_output

    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)
        self.emotion_transfer.reset_window_size(window_size)

    def get_model_name(self):
        return 'FlowReact'

    def get_version(self):
        return __version__
