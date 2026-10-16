import torch
import torch.nn as nn
from torch import Tensor
from .BasicBlock import ConvBlock, PositionalEncoding, init_biased_mask


def lengths_to_mask(lengths,
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask



class Intra3DMMEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, depth=4):
        super().__init__()
        # 1) 三种 token
        self.emb_exp   = nn.Linear(52, d_model)
        self.emb_rot   = nn.Linear(3,  d_model)
        self.emb_trans = nn.Linear(3,  d_model)

        # 2) Transformer（组内/组间混合）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.layers = nn.ModuleList([layer for _ in range(depth)])

        # 3) 帧级 pool：3 token → 1 token
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):                   # x: B×L×58
        # print(f"Intra3DMMEncoder input shape: {x.shape}")
        B, L, _ = x.shape
        e = self.emb_exp(x[..., :52])
        r = self.emb_rot(x[..., 52:55])
        t = self.emb_trans(x[..., 55:58])
        # 按帧拼：B×L×3×d  → B×L×d
        z = torch.stack([e, r, t], dim=2).mean(dim=2)
        for layer in self.layers:
            z = layer(z)
        return self.out(z)                  # B×L×d_model


class CrossFusion(nn.Module):
    def __init__(self, d_model=512, n_heads=8, depth=2):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.v2a = nn.ModuleList([layer for _ in range(depth)])
        self.a2v = nn.ModuleList([layer for _ in range(depth)])

    def forward(self, z_vis, z_aud):
        # 双向交叉
        for v2a, a2v in zip(self.v2a, self.a2v):
            z_vis = a2v(z_vis, z_aud)
            z_aud = v2a(z_aud, z_vis)
        return z_vis + z_aud                # 也可 concat

class EmotionExtractor(nn.Module):
    """
    输入:  z  (B, L, d_model)   来自 SpeakerEncoder 的输出
    输出:  e  (B, e_dim)        全局情感向量
           z  (B, L, d_model)   原序列（不做修改，直接透传，方便后续局部条件）
    """
    def __init__(self,
                 d_model: int = 256,
                 e_dim: int   = 128,
                 n_heads: int = 4):
        super().__init__()
        # 1) 可学习的 Query，用于 Attention Pooling
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

        # 2) 多头注意力：Query 与整段序列做加权平均
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # 3) 投影到情感维度
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, e_dim)
        )

    def forward(self, z):
        """
        z: (B, L, d_model)
        returns:
            e: (B, e_dim)  全局情感向量
            z: (B, L, d_model) 原序列（不做梯度截断，直接透传）
        """
        B, L, _ = z.shape
        q = self.query.expand(B, -1, -1)        # (B, 1, d_model)

        # Attention Pooling：Query 与整段序列交互
        e, _ = self.attn(q, z, z)               # (B, 1, d_model)
        e = e.squeeze(1)                        # (B, d_model)
        e = self.fc(e)                          # (B, e_dim)

        return e                             # e 全局情感, z 局部细节


class VAEModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int = 256,
                 **kwargs) -> None:
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, latent_dim)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=4,
                                                             dim_feedforward=latent_dim * 2,
                                                             dropout=0.1)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=1)
        self.mu_token = nn.Parameter(torch.randn(latent_dim))
        self.logvar_token = nn.Parameter(torch.randn(latent_dim))

    def forward(self, input):
        x = self.linear(input)  # B T D
        B, T, D = input.shape

        lengths = [len(item) for item in input]

        mu_token = torch.tile(self.mu_token, (B,)).reshape(B, 1, -1)
        logvar_token = torch.tile(self.logvar_token, (B,)).reshape(B, 1, -1)

        x = torch.cat([mu_token, logvar_token, x], dim=1)

        x = x.permute(1, 0, 2)

        token_mask = torch.ones((B, 2), dtype=bool, device=input.get_device())
        mask = lengths_to_mask(lengths, input.get_device())

        aug_mask = torch.cat((token_mask, mask), 1)

        x = self.seqTransEncoder(x, src_key_padding_mask=~aug_mask)

        mu = x[0]
        logvar = x[1]
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        motion_sample = self.sample_from_distribution(dist).to(input.get_device())
        return motion_sample, dist

    def sample_from_distribution(self, distribution):
        return distribution.rsample()


class Decoder(nn.Module):
    def __init__(self, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128, device='cpu', max_seq_len=751,
                 n_head=4, window_size=8, online=False):
        super(Decoder, self).__init__()

        self.feature_dim = feature_dim
        self.output_emotion_dim = 128
        self.window_size = window_size
        self.device = device
        self.online = online

        self.emo_extractor = EmotionExtractor(d_model=feature_dim, e_dim=self.output_emotion_dim, n_heads=4)

        self.vae_model = VAEModel(feature_dim + self.output_emotion_dim, feature_dim)

        if self.online:
            self.lstm = nn.LSTM(feature_dim, feature_dim, 1, batch_first=True)
            self.linear_3d = nn.Linear(output_3dmm_dim, feature_dim)
            self.linear_reaction = nn.Linear(feature_dim, feature_dim)
            decoder_layer_3d = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4, dim_feedforward=2 * feature_dim,
                                                          batch_first=True)
            self.listener_reaction_decoder_3d = nn.TransformerDecoder(decoder_layer_3d, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=2 * feature_dim,
                                                   batch_first=True)
        self.listener_reaction_decoder_1 = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.listener_reaction_decoder_2 = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.biased_mask = init_biased_mask(n_head=n_head, max_seq_len=max_seq_len, period=max_seq_len)

        self.listener_reaction_3dmm_map_layer = nn.Linear(feature_dim, output_3dmm_dim)
        self.listener_reaction_emotion_map_layer = nn.Sequential(
            nn.Linear(feature_dim + output_3dmm_dim, feature_dim),
            nn.Linear(feature_dim, output_emotion_dim)
        )
        self.PE = PositionalEncoding(feature_dim)

    def forward(self, encoded_feature, past_reaction_3dmm=None, past_reaction_emotion=None):
        e = self.emo_extractor(encoded_feature)  # (B, e_dim)
        # print(f"encoded_feature shape: {encoded_feature.shape}, encoded_feature_emo shape: {encoded_feature_emo.shape}")
        # encoded_feature shape: torch.Size([4, 16, 512]), encoded_feature_emo shape: torch.Size([4, 1, 128])

        B, TS = encoded_feature.shape[0], encoded_feature.shape[1]

        e = e.unsqueeze(1).expand(B, TS, -1)
        # concat
        encoded_feature = torch.cat((encoded_feature, e), dim=-1)

        if self.online:
            TL = self.window_size
        else:
            TL = TS
        motion_sample, dist = self.vae_model(encoded_feature)
        time_queries = torch.zeros(B, TL, self.feature_dim, device=encoded_feature.get_device())
        time_queries = self.PE(time_queries)
        tgt_mask = self.biased_mask[:, :TL, :TL].clone().detach().to(device=self.device).repeat(B, 1, 1)

        listener_reaction = self.listener_reaction_decoder_1(tgt=time_queries, memory=motion_sample.unsqueeze(1),
                                                             tgt_mask=tgt_mask)
        listener_reaction = self.listener_reaction_decoder_2(listener_reaction, listener_reaction, tgt_mask=tgt_mask)

        if self.online and (past_reaction_3dmm is not None):
            past_reaction_3dmm = self.linear_3d(past_reaction_3dmm)
            past_reaction_3dmm_last = past_reaction_3dmm[:, -1]

            tgt_mask = self.biased_mask[:, :(TL + past_reaction_3dmm.shape[1]),
                       :(TL + past_reaction_3dmm.shape[1])].detach().to(device=self.device).repeat(B, 1, 1)
            all_3dmm = torch.cat((past_reaction_3dmm, self.linear_reaction(listener_reaction)), dim=1)
            listener_3dmm_out = self.listener_reaction_decoder_3d(all_3dmm, all_3dmm, tgt_mask=tgt_mask)
            frame_num = listener_3dmm_out.shape[1]
            listener_3dmm_out = listener_3dmm_out[:, (frame_num - TL):]

            listener_3dmm_out, _ = self.lstm(listener_3dmm_out,
                                             (past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous(),
                                              past_reaction_3dmm_last.view(1, B, self.feature_dim).contiguous()))

            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_3dmm_out)
        else:
            listener_3dmm_out = self.listener_reaction_3dmm_map_layer(listener_reaction)

        listener_emotion_out = self.listener_reaction_emotion_map_layer(
            torch.cat((listener_3dmm_out, listener_reaction), dim=-1))

        return listener_3dmm_out, listener_emotion_out, dist

    def reset_window_size(self, window_size):
        self.window_size = window_size


class SpeakerBehaviourEncoder(nn.Module):
    def __init__(self, _3dmm_dim=224, audio_dim=78, feature_dim=128, device='cpu'):
        super(SpeakerBehaviourEncoder, self).__init__()

        self._3dmm_dim = _3dmm_dim
        self.audio_dim = audio_dim
        self.feature_dim = feature_dim
        self.device = device

        # self.video_encoder = VideoEncoder(img_size=img_size, feature_dim=feature_dim, device=device)
        # self._3dmm_encoder = nn.Linear(self._3dmm_dim, feature_dim)  # Assuming 3DMM has 58 dimensions
        self._3dmm_encoder = Intra3DMMEncoder(feature_dim,n_heads=8, depth=4)  # Assuming 3DMM has 58 dimensions
        self.audio_feature_map = nn.Linear(self.audio_dim, self.feature_dim)
        self.fusion_layer = CrossFusion(feature_dim, n_heads=4, depth=2)
        # self.fusion_layer = nn.Linear(self.feature_dim * 2, self.feature_dim)

    def forward(self, _3dmm, audio):
        video_feature = self._3dmm_encoder(_3dmm)
        audio_feature = self.audio_feature_map(audio)
        # speaker_behaviour_feature = self.fusion_layer(torch.cat((video_feature, audio_feature), dim=-1))
        speaker_behaviour_feature = self.fusion_layer(video_feature, audio_feature)

        return speaker_behaviour_feature


class TransformerVAE(nn.Module):
    def __init__(self, img_size=224, audio_dim=78, output_3dmm_dim=58, output_emotion_dim=25, feature_dim=128,
                 seq_len=750, task='online', window_size=8, device='cuda', **kwargs):
        super(TransformerVAE, self).__init__()

        self.img_size = img_size
        self.feature_dim = feature_dim
        self.output_3dmm_dim = output_3dmm_dim
        self.output_emotion_dim = output_emotion_dim
        self.seq_len = seq_len
        self.online = True if task == 'online' else False
        self.window_size = window_size
        self.device = device

        self.speaker_behaviour_encoder = SpeakerBehaviourEncoder(output_3dmm_dim, audio_dim, feature_dim, device)
        self.reaction_decoder = Decoder(output_3dmm_dim=output_3dmm_dim, output_emotion_dim=output_emotion_dim,
                                        feature_dim=feature_dim, device=device, window_size=self.window_size,
                                        online=self.online)
        self.fusion = nn.Linear(feature_dim + self.output_3dmm_dim + self.output_emotion_dim, feature_dim)

    def forward(self, speaker_3dmm=None, speaker_audio=None, **kwargs):
        distribution = []

        if self.online:
            speaker_3dmm = torch.stack(speaker_3dmm, dim=0).to(device=self.device)
            speaker_audio = torch.stack(speaker_audio, dim=0).to(device=self.device)

            # print(f"speaker_3dmm shape: {speaker_3dmm.shape}, speaker_audio shape: {speaker_audio.shape}")

            motion_lengths = (
                torch.as_tensor(
                    kwargs.get('motion_lengths', [self.seq_len] * len(speaker_3dmm)), device=self.device,
                ).clamp(max=self.seq_len)
            )

            frame_num = speaker_3dmm.shape[1]
            period = frame_num // self.window_size
            # num_windows = motion_lengths // self.window_size
            # motion_lengths Tensor([58, 720, 625, 750, ...]) ==> num_windows Tensor([7, 90, 78, 93, ...])

            reaction_3dmm = torch.zeros((speaker_3dmm.size(0), self.window_size, self.output_3dmm_dim),
                                        device=self.device)
            reaction_emotion = torch.zeros((speaker_3dmm.size(0), self.window_size, self.output_emotion_dim),
                                           device=self.device)

            for i in range(0, period):
                # mask = (~(num_windows < i)).view(-1, 1).repeat(1, self.window_size)
                # mask = mask.unsqueeze(-1).to(device=speaker_video)
                speaker_3dmm_, speaker_audio_ = (speaker_3dmm[:, :(i + 1) * self.window_size],
                                                  speaker_audio[:, :(i + 1) * self.window_size])
                encoded_feature = self.speaker_behaviour_encoder(speaker_3dmm_, speaker_audio_)

                # modality fusion
                encoded_feature = self.fusion(
                    torch.cat((encoded_feature, reaction_3dmm, reaction_emotion), dim=-1))

                if i != 0:
                    past_reaction_3dmm, past_reaction_emotion = (reaction_3dmm[:, :i * self.window_size],
                                                                 reaction_emotion[:, :i * self.window_size])
                    current_reaction_3dmm, current_reaction_emotion = (reaction_3dmm[:, i * self.window_size:],
                                                                       reaction_emotion[:, i * self.window_size:])
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature,
                                                                                          past_reaction_3dmm)

                    reaction_3dmm = torch.cat(
                        (past_reaction_3dmm, listener_3dmm_out, current_reaction_3dmm), dim=1)
                    reaction_emotion = torch.cat(
                        (past_reaction_emotion, listener_emotion_out, current_reaction_emotion), dim=1)

                else:
                    listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
                    reaction_3dmm = torch.cat((listener_3dmm_out, reaction_3dmm), dim=1)
                    reaction_emotion = torch.cat((listener_emotion_out, reaction_emotion), dim=1)

                distribution.append(dist)

            listener_3dmm_out, listener_emotion_out = reaction_3dmm[:, :frame_num], reaction_emotion[:, :frame_num]
            seq_mask = lengths_to_mask(lengths=motion_lengths,
                                       device=speaker_3dmm.device,
                                       max_len=frame_num).unsqueeze(-1).float()
            listener_3dmm_out = listener_3dmm_out * seq_mask
            listener_emotion_out = listener_emotion_out * seq_mask

            listener_3dmm_out, listener_emotion_out = \
                list(listener_3dmm_out), list(listener_emotion_out)
            return listener_3dmm_out, listener_emotion_out, distribution

        else:
            listener_3dmm_outs = []
            listener_emotion_outs = []
            for speaker_3dmm_, speaker_audio_ in zip(speaker_3dmm, speaker_audio):  # motion_lengths
                speaker_3dmm_, speaker_audio_ = \
                    speaker_3dmm_.unsqueeze(0).to(self.device), speaker_audio_.unsqueeze(0).to(self.device)
                encoded_feature = self.speaker_behaviour_encoder(speaker_3dmm_, speaker_audio_)
                listener_3dmm_out, listener_emotion_out, dist = self.reaction_decoder(encoded_feature)
                listener_3dmm_outs.append(listener_3dmm_out.squeeze(0))
                listener_emotion_outs.append(listener_emotion_out.squeeze(0))
                distribution.append(dist)

            return listener_3dmm_outs, listener_emotion_outs, distribution

    def reset_window_size(self, window_size):
        self.window_size = window_size
        self.reaction_decoder.reset_window_size(window_size)

    def get_model_name(self):
        return 'TransformerVAE'
