import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4000):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):
        """
        x: (B, T, D)
        offset: starting time index in the global sequence
        """
        return x + self.pe[offset: offset + x.size(1)]




class TemporalTransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 n_layers=4,
                 n_heads=4,
                 dropout=0.1,
                 window_size=16):
        super().__init__()

        self.max_window_size = window_size

        self.window_size = window_size

        self.training = True

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_enc = SinusoidalPositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        return: (B, T, latent_dim)
        """
        B, T, _ = x.shape
        if self.training:
            W = torch.randint(8, self.max_window_size + 1, (1,)).item()
        else:
            W = self.window_size

        x = self.input_proj(x)

        outputs = []
        for start in range(0, T, W):
            end = min(start + W, T)

            chunk = x[:, start:end]                    # (B, w, D)
            chunk = self.pos_enc(chunk, offset=start)  # ✅ global time
            z = self.encoder(chunk)                    # (B, w, D)

            outputs.append(z)

        z = torch.cat(outputs, dim=1)                  # (B, T, D)
        return self.norm(z)

    def set_training(self):
        self.training = True

    def set_inference(self):
        self.training = False

    def reset_window_size(self, window_size):
        self.window_size = window_size



class TemporalMLPDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim)
        )

    def forward(self, z):
        """
        z: (B, T, latent_dim)
        return: (B, T, out_dim)
        """
        return self.net(z)



class Emotion3DMMTransformer(nn.Module):
    """
    This module is designed as a reusable pretrained functional block.

    External models should ONLY call:
        - to_3dmm(emotion)
        - to_emotion(params)
    """

    def __init__(self,
                 emotion_dim=25,
                 param_dim=58,
                 latent_dim=128,
                 n_layers=4,
                 n_heads=4,
                 window_size=16,
                 **kwargs):
        super().__init__()

        # ========== Encoders ==========
        self.emotion_encoder = TemporalTransformerEncoder(
            input_dim=emotion_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size
        )

        self.param_encoder = TemporalTransformerEncoder(
            input_dim=param_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            window_size=window_size
        )

        # ========== Decoders ==========
        self.emotion_decoder = TemporalMLPDecoder(
            latent_dim=latent_dim,
            out_dim=emotion_dim
        )

        self.param_decoder = TemporalMLPDecoder(
            latent_dim=latent_dim,
            out_dim=param_dim
        )

    # ============================================================
    # Emotion → 3DMM
    # ============================================================
    def to_3dmm(self, emotion, **kwargs):
        """
        emotion: (B, T, emotion_dim)

        return:
            params: (B, T, param_dim)
            latent: (B, T, latent_dim)
        """
        z = self.emotion_encoder(emotion)
        params = self.param_decoder(z)
        return params, z

    # ============================================================
    # 3DMM → Emotion
    # ============================================================
    def to_emotion(self, params, **kwargs):
        """
        params: (B, T, param_dim)

        return:
            emotion: (B, T, emotion_dim)
            latent: (B, T, latent_dim)
        """
        z = self.param_encoder(params)
        emotion = self.emotion_decoder(z)
        return emotion, z

    def forward(self, emotion, params, **kwargs):
        """
        emotion: (B, T, emotion_dim)
        params: (B, T, param_dim)

        return:
            pred_3dmm: (B, T, param_dim)
            pred_emotion: (B, T, emotion_dim)
            z_e: (B, T, latent_dim)
            z_p: (B, T, latent_dim)
        """
        pred_3dmm, z_e = self.to_3dmm(emotion)
        pred_emotion, z_p = self.to_emotion(params)

        return {
            "pred_3dmm": pred_3dmm,
            "pred_emotion": pred_emotion,
            "z_e": z_e,
            "z_p": z_p
        }

    def set_training(self):
        self.emotion_encoder.set_training()
        self.param_encoder.set_training()

    def set_inference(self):
        self.emotion_encoder.set_inference()
        self.param_encoder.set_inference()

    def reset_window_size(self, window_size):
        self.emotion_encoder.reset_window_size(window_size)
        self.param_encoder.reset_window_size(window_size)

    def get_model_name(self):
        return "Emotion3DMMTransformer"
