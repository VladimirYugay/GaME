""" This module contains utility functions for rendering performance evaluation. """
import os
from pathlib import Path

import cv2
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from src.flashsplat.gaussian_renderer import flashsplat_render
from src.utils import utils


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """ Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: The first image.
        img2: The second image.
    Returns:
        The PSNR value.
    """
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate_all_rendering(gaussian_model, dataset, output_path: Path = None,
                           refinement_state=None, set_info=None, frame_rate=5) -> tuple:
    """ Evaluate the rendering quality of a Gaussian model against a dataset
    Args:
        gaussian_model: The Gaussian model.
        dataset: The dataset to evaluate against.
        output_path: The output path to save the rendered images.
    Returns:
        psnr_values: The PSNR values for all the keyframes.
        lpips: The LPIPS values for all the keyframes.
        ssim: The SSIM values for all the keyframes.
        depth_l1s: The depth L1 values for all the keyframes.
    """
    with torch.no_grad():
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).cuda()
        background = torch.zeros(3).cuda()
        pipe = utils.flashsplat_pipe()

        psnr_values, lpips_values, ssim_values, l1_values = [], [], [], []
        num_frames = len(dataset)
        eval_start = 0  # dataset.start_frame
        video_path = output_path / \
            f"{refinement_state}{set_info}_in_progress.mp4"
        for frame_id in tqdm(range(eval_start, num_frames), "Evaluating frames"):

            sample = dataset[frame_id]
            color, depth = sample["color"], sample["depth"]
            pose, intrinsics = sample["pose"], sample["intrinsics"]

            frame_tensors = {
                "color": utils.np2torch(color, device="cuda").permute(2, 0, 1) / 255.0,
                "depth": utils.np2torch(depth, device="cuda"),
                "pose": utils.np2torch(pose, device="cuda"),
                "intrinsics": intrinsics,
            }

            # frame = utils.dict2device(frame_tensors, "cuda")
            frame = frame_tensors
            gt_color, gt_depth = frame["color"], frame["depth"]
            pose, intrinsics = frame["pose"], frame["intrinsics"]
            flashsplat_view = utils.flashsplat_cam(
                gt_color, gt_depth, None, intrinsics, pose.cpu(), frame_id)
            render_pkg = flashsplat_render(
                flashsplat_view, gaussian_model, pipe, background, obj_num=256)
            color, depth = render_pkg["render"], render_pkg["depth"]
            # del render_pkg
            # torch.cuda.empty_cache()
            rendered_color = torch.clamp(color, 0.0, 1.0)

            if frame_id == eval_start:
                # Read the first image to get the dimensions
                _, height, width = color.shape

                # Define the codec and create VideoWriter object
                # You can use other codecs like 'XVID'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(
                    video_path, fourcc, frame_rate, (width, height))

            video_frame = cv2.cvtColor((color.cpu().clip(0.0, 1.0).permute(
                (1, 2, 0)) * 255).to(torch.uint8).numpy(), cv2.COLOR_RGB2BGR)
            video.write(video_frame)
            psnr_values.append(
                calc_psnr(rendered_color, gt_color).mean().item())
            lpips_values.append(lpips_model(
                rendered_color[None], gt_color[None]).mean().item())
            ssim_values.append(
                ms_ssim(rendered_color[None], gt_color[None], data_range=1.0).item())
            l1_values.append(torch.mean(torch.abs(depth - gt_depth)).item())

        video.release()
        psnr, lpips, ssim, l1 = torch.Tensor(psnr_values).mean(
        ), torch.Tensor(lpips_values).mean(
        ), torch.Tensor(ssim_values).mean(
        ), torch.Tensor(l1_values).mean()
        final_video_name = f"{refinement_state}{set_info}_psnr_{psnr:.03f}_lpips_{lpips:.03f}_ssim_{ssim:.03f}_l1_{l1:.03f}.mp4"
        new_video_path = output_path / final_video_name
        os.rename(video_path, new_video_path)
        print("Video saved at", new_video_path)

        return psnr.item(), lpips.item(), ssim.item(), l1.item()
