import math
import os
import numpy as np
import torch
import hydra
from matplotlib import pyplot as plt
from torchvision import transforms
import cv2
from utils.util import torch_img_to_np, _fix_image, torch_img_to_np2, set_seed
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator
from skimage.io import imsave
import skvideo.io


def obtain_seq_index(index, num_frames, semantic_radius=13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)
    return semantic.transpose(1, 2)


class Render(object):
    """Computes and stores the average and current value"""

    def __init__(self, device='cpu', **kwargs):
        dir = hydra.utils.to_absolute_path("external/FaceVerse")
        self.faceverse, _ = get_faceverse(
            path=os.path.join(dir, "data/faceverse_simple_v2.npy"),
            device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = torch.from_numpy(np.load(os.path.join(dir, "reference_full.npy"))).float().view(1, -1)[:, :150]
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()
        ckpt_path = hydra.utils.to_absolute_path("external/PIRender/cur_model_fold.pth")
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"No checkpoint found at {ckpt_path}")
        self.pi_render.load_state_dict(torch.load(ckpt_path))

        self.mean_face = torch.FloatTensor(
            np.load(os.path.join(dir, "mean_face.npy")).astype(np.float32)).view(1, 1, -1).to(device)
        self.std_face = torch.FloatTensor(
            np.load(os.path.join(dir, "std_face.npy")).astype(np.float32)).view(1, 1, -1).to(device)

        transform_reverse = kwargs.get('transform_reverse', 'zero_center')
        if transform_reverse == 'zero_center':
            self._reverse_transform_3dmm = transforms.Lambda(lambda e: e + self.mean_face)
        elif transform_reverse == 'standard':
            self._reverse_transform_3dmm = transforms.Lambda(lambda e: e * self.std_face + self.mean_face)
        else:
            raise ValueError(f"Unknown transform_reverse: {transform_reverse}")
        self._transform = transforms.Lambda(
            lambda e: (lambda tmp: tmp.__setitem__((slice(None), -1), e[:, -1] - self.mean_face[0, 0, -1]) or tmp)(e.clone()))

    def rendering(self, path, ind, listener_vectors, speaker_video_clip, listener_reference, listener_video_clip):
        if len(listener_vectors.shape) > 2:
            listener_vectors = listener_vectors.squeeze(0)

        # 3D video
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]
        listener_vectors = self._transform(listener_vectors)
        # print(f"maximum of listener_3dmm_out: {torch.max(listener_vectors)}")
        # print(f"minimum of listener_3dmm_out: {torch.min(listener_vectors)}")

        T_unit = 512
        rendered_img_r_list = []
        for i in range(math.ceil(T / T_unit)):
            if i != math.ceil(T / T_unit) - 1:
                listener_vectors_i = listener_vectors[i * T_unit:(i + 1) * T_unit]
            else:
                listener_vectors_i = listener_vectors[i * T_unit:]

            self.faceverse.batch_size = listener_vectors_i.shape[0]
            self.faceverse.init_coeff_tensors()
            exp_tensor = listener_vectors_i[:, :52].to(listener_vectors_i.get_device())
            rot_tensor = listener_vectors_i[:, 52:55].to(listener_vectors_i.get_device())
            trans_tensor = listener_vectors_i[:, 55:].to(listener_vectors_i.get_device())
            self.faceverse.exp_tensor = exp_tensor
            self.faceverse.rot_tensor = rot_tensor
            self.faceverse.trans_tensor = trans_tensor
            self.faceverse.id_tensor = self.id_tensor.reshape(1, 150).repeat(
                listener_vectors_i.shape[0], 1).to(listener_vectors_i.get_device())

            pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
            rendered_img_r = pred_dict['rendered_img']
            rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
            rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)
            rendered_img_r_list.append(rendered_img_r)
        rendered_img_r = np.concatenate(rendered_img_r_list, axis=0)

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).reshape(
            listener_vectors.shape[0], C, H, W)

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            with torch.no_grad():
                output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (672, 224))
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

        listener_video_clip = torch_img_to_np2(listener_video_clip)  # [L, ...]
        path_real = os.path.join(path, ind, 'real')
        if not os.path.exists(path_real):
            os.makedirs(path_real)
        path_fake = os.path.join(path, ind, 'fake')
        if not os.path.exists(path_fake):
            os.makedirs(path_fake)
        path_speaker = os.path.join(path, ind, 'speaker')
        if not os.path.exists(path_speaker):
            os.makedirs(path_speaker)

        # n_fake = listener_videos.shape[0] if hasattr(listener_videos, 'shape') else len(listener_videos)
        # n_real = listener_video_clip.shape[0] if hasattr(listener_video_clip, 'shape') else len(listener_video_clip)
        # n_speaker = speaker_video_clip.shape[0] if hasattr(speaker_video_clip, 'shape') else len(speaker_video_clip)
        for i in range(0, rendered_img_r.shape[0], 30):
            # if i < n_fake and i < n_real:
            cv2.imwrite(os.path.join(path_fake, 'img_' + str(i + 1) + '.png'), listener_videos[i])
            cv2.imwrite(os.path.join(path_real, 'img_' + str(i + 1) + '.png'), listener_video_clip[i])
            cv2.imwrite(os.path.join(path_speaker, 'img_' + str(i + 1) + '.png'), speaker_video_clip[i])

    def rendering_for_test(self, path, ind, listener_vectors, speaker_video_clip, listener_reference, listener_video_clip):
        if len(listener_vectors.shape) > 2:
            listener_vectors = listener_vectors.squeeze(0)

        # 3D video
        print("---------------------------1---------------------------")
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]
        print("---------------------------2---------------------------")
        listener_vectors = self._transform(listener_vectors)
        print("---------------------------3---------------------------")
        # print(f"maximum of listener_3dmm_out: {torch.max(listener_vectors)}")
        # print(f"minimum of listener_3dmm_out: {torch.min(listener_vectors)}")

        # T_unit = 512
        # rendered_img_r_list = []
        # for i in range(math.ceil(T / T_unit)):
        #     if i != math.ceil(T / T_unit) - 1:
        #         listener_vectors_i = listener_vectors[i * T_unit:(i + 1) * T_unit]
        #     else:
        #         listener_vectors_i = listener_vectors[i * T_unit:]

            # self.faceverse.batch_size = listener_vectors_i.shape[0]
            # self.faceverse.init_coeff_tensors()
            # exp_tensor = listener_vectors_i[:, :52].to(listener_vectors_i.get_device())
            # rot_tensor = listener_vectors_i[:, 52:55].to(listener_vectors_i.get_device())
            # trans_tensor = listener_vectors_i[:, 55:].to(listener_vectors_i.get_device())
            # self.faceverse.exp_tensor = exp_tensor
            # self.faceverse.rot_tensor = rot_tensor
            # self.faceverse.trans_tensor = trans_tensor
            # self.faceverse.id_tensor = self.id_tensor.reshape(1, 150).repeat(
            #     listener_vectors_i.shape[0], 1).to(listener_vectors_i.get_device())

            # pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        #     rendered_img_r = pred_dict['rendered_img']
        #     rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        #     rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)
        #     rendered_img_r_list.append(rendered_img_r)
        # rendered_img_r = np.concatenate(rendered_img_r_list, axis=0)

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).reshape(
            listener_vectors.shape[0], C, H, W)

        print("---------------------------4---------------------------")

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            with torch.no_grad():
                output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        print("---------------------------5---------------------------")

        listener_videos = np.concatenate(output_dict_list, axis=0)
        print("---------------------------6---------------------------")
        # speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (224, 224))
        for i in range(listener_videos.shape[0]):
            combined_img = np.zeros((224, 224, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = listener_videos[i]
            out.write(combined_img)
        out.release()

        print("---------------------------7---------------------------")

        # listener_video_clip = torch_img_to_np2(listener_video_clip)  # [L, ...]
        # path_real = os.path.join(path, ind, 'real')
        # if not os.path.exists(path_real):
        #     os.makedirs(path_real)
        # path_fake = os.path.join(path, ind, 'fake')
        # if not os.path.exists(path_fake):
        #     os.makedirs(path_fake)
        # path_speaker = os.path.join(path, ind, 'speaker')
        # if not os.path.exists(path_speaker):
        #     os.makedirs(path_speaker)

        # n_fake = listener_videos.shape[0] if hasattr(listener_videos, 'shape') else len(listener_videos)
        # n_real = listener_video_clip.shape[0] if hasattr(listener_video_clip, 'shape') else len(listener_video_clip)
        # n_speaker = speaker_video_clip.shape[0] if hasattr(speaker_video_clip, 'shape') else len(speaker_video_clip)
        # for i in range(0, rendered_img_r.shape[0], 30):
        #     # if i < n_fake and i < n_real:
        #     cv2.imwrite(os.path.join(path_fake, 'img_' + str(i + 1) + '.png'), listener_videos[i])
        #     cv2.imwrite(os.path.join(path_real, 'img_' + str(i + 1) + '.png'), listener_video_clip[i])
        #     cv2.imwrite(os.path.join(path_speaker, 'img_' + str(i + 1) + '.png'), speaker_video_clip[i])


    def rendering_for_sample(self, path, ind, listener_vectors, speaker_video_clip, listener_reference, listener_video_clip):
        if len(listener_vectors.shape) > 2:
            listener_vectors = listener_vectors.squeeze(0)

        # 3D video
        print("---------------------------1---------------------------")
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]
        print("---------------------------2---------------------------")
        listener_vectors = self._transform(listener_vectors)
        print("---------------------------3---------------------------")
        # print(f"maximum of listener_3dmm_out: {torch.max(listener_vectors)}")
        # print(f"minimum of listener_3dmm_out: {torch.min(listener_vectors)}")

        # T_unit = 512
        # rendered_img_r_list = []
        # for i in range(math.ceil(T / T_unit)):
        #     if i != math.ceil(T / T_unit) - 1:
        #         listener_vectors_i = listener_vectors[i * T_unit:(i + 1) * T_unit]
        #     else:
        #         listener_vectors_i = listener_vectors[i * T_unit:]

            # self.faceverse.batch_size = listener_vectors_i.shape[0]
            # self.faceverse.init_coeff_tensors()
            # exp_tensor = listener_vectors_i[:, :52].to(listener_vectors_i.get_device())
            # rot_tensor = listener_vectors_i[:, 52:55].to(listener_vectors_i.get_device())
            # trans_tensor = listener_vectors_i[:, 55:].to(listener_vectors_i.get_device())
            # self.faceverse.exp_tensor = exp_tensor
            # self.faceverse.rot_tensor = rot_tensor
            # self.faceverse.trans_tensor = trans_tensor
            # self.faceverse.id_tensor = self.id_tensor.reshape(1, 150).repeat(
            #     listener_vectors_i.shape[0], 1).to(listener_vectors_i.get_device())

            # pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        #     rendered_img_r = pred_dict['rendered_img']
        #     rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        #     rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)
        #     rendered_img_r_list.append(rendered_img_r)
        # rendered_img_r = np.concatenate(rendered_img_r_list, axis=0)

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        print("duration:", duration)
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).reshape(
            listener_vectors.shape[0], C, H, W)

        print("---------------------------4---------------------------")

        for i in range(20):
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            with torch.no_grad():
                output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)
            print("Processed segment ", i)

        print("---------------------------5---------------------------")

        listener_videos = np.concatenate(output_dict_list, axis=0)
        print("---------------------------6---------------------------")
        # speaker_video_clip = torch_img_to_np2(speaker_video_clip)
        # 将speaker_video_clip的torch_img_to_np2转换分段执行，防止显存不足
        speaker_video_clip_list = []
        segment_length = 50  # 每个段的长度
        num_segments = math.ceil(speaker_video_clip.shape[0] / segment_length)
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, speaker_video_clip.shape[0])
            speaker_video_clip_segment = speaker_video_clip[start_idx:end_idx]
            speaker_video_clip_np = torch_img_to_np2(speaker_video_clip_segment)
            speaker_video_clip_list.append(speaker_video_clip_np)
            print("Processed speaker segment ", i)


        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (448, 224))
        for i in range(listener_vectors.shape[0]):
            combined_img = np.zeros((224, 448, 3), dtype=np.uint8)
            # combined_img[0:224, 0:224] = speaker_video_clip[i]
            # # combined_img[0:224, 224:448] = rendered_img_r[i]
            # combined_img[0:224, 224:] = listener_videos[i]

            # Convert CHW -> HWC if needed
            speaker_frame = speaker_video_clip[i]
            if speaker_frame.shape[0] == 3:  # CHW detected
                speaker_frame = np.transpose(speaker_frame, (1, 2, 0))
            combined_img[0:224, 0:224] = speaker_frame

            listener_frame = listener_videos[i]
            if listener_frame.shape[0] == 3:  # CHW detected
                listener_frame = np.transpose(listener_frame, (1, 2, 0))
            combined_img[0:224, 224:] = listener_frame
            out.write(combined_img)
        out.release()

        print("---------------------------7---------------------------")

        # listener_video_clip = torch_img_to_np2(listener_video_clip)  # [L, ...]
        # path_real = os.path.join(path, ind, 'real')
        # if not os.path.exists(path_real):
        #     os.makedirs(path_real)
        # path_fake = os.path.join(path, ind, 'fake')
        # if not os.path.exists(path_fake):
        #     os.makedirs(path_fake)
        # path_speaker = os.path.join(path, ind, 'speaker')
        # if not os.path.exists(path_speaker):
        #     os.makedirs(path_speaker)

        # n_fake = listener_videos.shape[0] if hasattr(listener_videos, 'shape') else len(listener_videos)
        # n_real = listener_video_clip.shape[0] if hasattr(listener_video_clip, 'shape') else len(listener_video_clip)
        # n_speaker = speaker_video_clip.shape[0] if hasattr(speaker_video_clip, 'shape') else len(speaker_video_clip)
        # for i in range(0, rendered_img_r.shape[0], 30):
        #     # if i < n_fake and i < n_real:
        #     cv2.imwrite(os.path.join(path_fake, 'img_' + str(i + 1) + '.png'), listener_videos[i])
        #     cv2.imwrite(os.path.join(path_real, 'img_' + str(i + 1) + '.png'), listener_video_clip[i])
        #     cv2.imwrite(os.path.join(path_speaker, 'img_' + str(i + 1) + '.png'), speaker_video_clip[i])


    def rendering_for_it(self, path, ind, listener_vectors, listener_reference):
        """
        只渲染listener视频，不包含speaker视频
        """
        if len(listener_vectors.shape) > 2:
            listener_vectors = listener_vectors.squeeze(0)

        # 3D video
        print("---------------------------1---------------------------")
        T = listener_vectors.shape[0]
        listener_vectors = self._reverse_transform_3dmm(listener_vectors)[0]
        print("---------------------------2---------------------------")
        listener_vectors = self._transform(listener_vectors)
        print("---------------------------3---------------------------")

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 20
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).reshape(
            listener_vectors.shape[0], C, H, W)

        print("---------------------------4---------------------------")

        for i in range(20):
            print("Processing segment ", i)
            if i != 19:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            with torch.no_grad():
                output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        print("---------------------------5---------------------------")

        listener_videos = np.concatenate(output_dict_list, axis=0)
        print("---------------------------6---------------------------")

        out = cv2.VideoWriter(os.path.join(path, ind + "_val.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (224, 224))
        for i in range(listener_videos.shape[0]):
            combined_img = np.zeros((224, 224, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = listener_videos[i]
            out.write(combined_img)
        out.release()

        print("---------------------------7---------------------------")