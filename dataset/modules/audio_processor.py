# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os
from contextlib import nullcontext

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor
from dataset.tools.util import resample_audio
from framework.feature_extractor.wav2vec import Wav2VecModel
import soundfile as sf
import torchaudio

torchaudio.set_audio_backend("sox_io")


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """

    def __init__(
            self,
            sample_rate,
            fps,
            wav2vec_model_path,
            only_last_features: bool = True,
            audio_separator_model_path: str = None,
            audio_separator_model_name: str = None,
            cache_dir: str = '',
            device="cuda:0",
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = torch.device(device)

        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_model_path,
            local_files_only=True,
        ).to(device=self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder.eval()
        self.only_last_features = only_last_features

        if audio_separator_model_name is not None:
            from audio_separator.separator import Separator
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator = None
            print("Use audio directly without vocals seperator.")

        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path,
                                                                                  local_files_only=True)

    def load_waveform(self, wav_file: str):
        """Load a waveform and resample it to the configured sample rate."""
        if self.audio_separator is not None:
            outputs = self.audio_separator.separate(wav_file)
            if len(outputs) <= 0:
                raise RuntimeError("Audio separate failed.")

            vocal_audio_file = outputs[0]
            vocal_audio_name, _ = os.path.splitext(vocal_audio_file)
            vocal_audio_file = os.path.join(self.audio_separator.output_dir, vocal_audio_file)
            vocal_audio_file = resample_audio(
                vocal_audio_file,
                os.path.join(self.audio_separator.output_dir, f"{vocal_audio_name}-16k.wav"),
                self.sample_rate,
            )
        else:
            vocal_audio_file = wav_file

        speech_array, sampling_rate = librosa.load(vocal_audio_file, sr=self.sample_rate)
        return speech_array, sampling_rate

    def prepare_waveform(
            self,
            speech_array,
            sampling_rate: int,
            clip_length: int = -1,
            seq_len: int = None,
            warn_length_mismatch: bool = True,
    ):
        """Normalize a waveform and convert it to a batched CPU tensor."""
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(
                speech_array,
                sampling_rate=sampling_rate,
            ).input_values
        )
        computed_seq_len = math.ceil(len(audio_feature) / self.sample_rate * self.fps)
        if seq_len is None:
            seq_len = computed_seq_len
        elif computed_seq_len != seq_len and warn_length_mismatch:
            print(
                f"Warning: seq_len {seq_len} is not equal to "
                f"computed version {computed_seq_len}"
            )
        audio_length = seq_len

        audio_feature = torch.from_numpy(audio_feature).float()
        if clip_length > 0 and seq_len % clip_length != 0:
            pad_frames = clip_length - seq_len % clip_length
            audio_feature = torch.nn.functional.pad(
                audio_feature,
                (0, pad_frames * (self.sample_rate // self.fps)),
                "constant",
                0.0,
            )
            seq_len += pad_frames

        return audio_feature.unsqueeze(0), seq_len, audio_length

    def encode_waveform(
            self,
            audio_feature: torch.Tensor,
            seq_len: int,
            return_cpu: bool = True,
            precision: str = "fp32",
    ):
        """Encode a normalized waveform into frame-aligned Wav2Vec features."""
        if precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision: {precision}")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 audio encoding requires a CUDA device")

        if not isinstance(audio_feature, torch.Tensor):
            audio_feature = torch.as_tensor(audio_feature, dtype=torch.float32)
        if audio_feature.ndim == 1:
            audio_feature = audio_feature.unsqueeze(0)
        audio_feature = audio_feature.to(device=self.device, dtype=torch.float32, non_blocking=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if precision == "fp16"
            else nullcontext()
        )
        with torch.inference_mode():
            with amp_context:
                embeddings = self.audio_encoder(
                    audio_feature,
                    seq_len=seq_len,
                    output_attentions=False,
                    output_hidden_states=not self.only_last_features,
                    verbose=False,
                )

        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze(0)
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.detach()
        return audio_emb.cpu() if return_cpu else audio_emb

    def preprocess(
            self,
            wav_file: str,
            clip_length: int = -1,
            seq_len: int = None,
            return_cpu: bool = True,
            precision: str = "fp32",
    ):
        """
        Preprocess a WAV audio file by separating the vocals from the background and resampling it to a 16 kHz sample rate.
        The separated vocal track is then converted into wav2vec2 for further processing or analysis.

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Raises:
            RuntimeError: Raises an exception if the WAV file cannot be processed. This could be due to issues
                        such as file not found, unsupported file format, or errors during the audio processing steps.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        speech_array, sampling_rate = self.load_waveform(wav_file)
        audio_feature, encoded_seq_len, audio_length = self.prepare_waveform(
            speech_array,
            sampling_rate,
            clip_length=clip_length,
            seq_len=seq_len,
        )
        audio_emb = self.encode_waveform(
            audio_feature,
            seq_len=encoded_seq_len,
            return_cpu=return_cpu,
            precision=precision,
        )
        return audio_emb, audio_length

    def get_embedding(self, wav_file: str):
        """preprocess wav audio file convert to embeddings

        Args:
            wav_file (str): The path to the WAV file to be processed. This file should be accessible and in WAV format.

        Returns:
            torch.tensor: Returns an audio embedding as a torch.tensor
        """
        audio_emb, _ = self.preprocess(wav_file)
        return audio_emb

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
