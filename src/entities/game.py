import random

import cv2 as cv
import numpy as np
import torch
from pytorch_msssim import ssim
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import wandb
from src.entities.arguments import ArgumentParser, OptimizationParams
from src.entities.losses import isotropic_loss, l1_loss
from src.flashsplat.gaussian_renderer import GaussianModel, flashsplat_render
from src.utils import io_utils, utils
from src.utils.mapping_eval import evaluate_all_rendering


class GaME(object):

    def __init__(self, config: dict, wandb_online: bool = False, checkpoint_path=None, all_train_data=None) -> None:
        """Initialise the GaME model.

        Args:
            config: SLAM config dict. Expected keys include ``num_label_channels``,
                ``scale`` (optional), and all optimisation/change-detection
                hyperparameters.
            wandb_online: Whether to log metrics to wandb.
            checkpoint_path: If provided, resume from this checkpoint file.
            all_train_data: List of training datasets required when resuming
                from a checkpoint to reload keyframe data.
        """
        self.wandb_online = wandb_online
        self.config = config
        self.keyframes = {}
        self.estimated_poses = {}
        self.occlusion_masks = {}
        self.num_label_channels = config["num_label_channels"]
        self.opt_params = OptimizationParams(ArgumentParser(description="lol"))
        self.ignored_frames = set()
        self._last_keyframe_id = None

        if checkpoint_path is not None:
            self.load(checkpoint_path, all_train_data)
        else:
            scale = self.config.get("scale", 1.0)
            self.gaussian_model = GaussianModel(3, scale=scale)
            self.gaussian_model.training_setup(self.opt_params)

    def _sample_valid_keyframe(self, selected_frames, only_frame_id):
        """Sample a keyframe that is not ignored and not fully occluded by occlusion masks.

        Randomly draws from ``selected_frames`` until a valid candidate is found.
        A frame is rejected and added to ``self.ignored_frames`` if any of its masks
        is covered above ``frame_ignore_thresh`` by the corresponding occlusion mask.

        Args:
            selected_frames: List of keyframe IDs to sample from.
            only_frame_id: If not ``None``, always returns this ID immediately.

        Returns:
            A valid keyframe ID.
        """
        if only_frame_id is not None:
            return only_frame_id

        thresh = self.config["occlusion_ignore_threshold"]
        active_frames = [f for f in selected_frames if f not in self.ignored_frames]
        keyframe_okay = False
        while not keyframe_okay:
            keyframe_id = random.choice(active_frames)
            if keyframe_id not in self.occlusion_masks:
                keyframe_okay = True
            else:
                all_masks = self.keyframes[keyframe_id]["masks"].cuda()
                ignore_mask = self.occlusion_masks[keyframe_id]
                covered_scores = (ignore_mask * all_masks).sum(dim=(1, 2)) / all_masks.sum(dim=(1, 2))
                if (covered_scores > thresh).any():
                    self.ignored_frames.add(keyframe_id)
                else:
                    keyframe_okay = True
        return keyframe_id

    def _densification_step(self, iteration, total_loss, visibility_filter, radii, viewspace_point_tensor):
        """Run one densification step during refinement.

        Logs metrics to wandb, updates max radii, accumulates densification
        statistics, and conditionally runs densify-and-prune or opacity reset.

        Args:
            iteration: Current iteration index.
            total_loss: Scalar loss tensor from the current forward pass.
            visibility_filter: Boolean tensor indicating which gaussians are visible.
            radii: Per-gaussian radii in image space.
            viewspace_point_tensor: Viewspace point tensor with gradients retained.
        """
        if self.wandb_online:
            wandb.log({"current loss": total_loss.item(),
                       "num_gaussians": self.gaussian_model.get_scaling.shape[0]})

        opacity_reset_interval = self.config["opacity_reset_interval"]
        densify_from_iter = self.config["densify_from_iter"]
        densification_interval = self.config["densification_interval"]
        densify_grad_threshold = self.config["densify_grad_threshold"]
        min_grad = self.config["min_grad"]

        coords = self.gaussian_model.get_xyz
        cameras_extent = (coords.max(0).values - coords.min(0).values).max()

        # keep track of max radii in image-space for pruning
        if self.gaussian_model.max_radii2D.shape[0] > 0:
            self.gaussian_model.max_radii2D[visibility_filter] = torch.max(
                self.gaussian_model.max_radii2D[visibility_filter], radii[visibility_filter])
        else:
            self.gaussian_model.max_radii2D = torch.zeros_like(visibility_filter)
            self.gaussian_model.max_radii2D[visibility_filter] = radii[visibility_filter]
        self.gaussian_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > densify_from_iter and iteration % densification_interval == 0:
            size_threshold = 20 if iteration > opacity_reset_interval else None
            self.gaussian_model.densify_and_prune(
                densify_grad_threshold, min_grad, cameras_extent, size_threshold)

        if iteration % opacity_reset_interval == 0 or iteration == densify_from_iter:
            self.gaussian_model.reset_opacity()

    def optimize_model(self, iterations=100, only_frame_id=None, refinement=False):
        """Optimise the Gaussian model for a fixed number of iterations.

        Samples keyframes, renders them, computes color/depth/regularisation
        losses, and back-propagates. In non-refinement mode, prunes low-opacity
        Gaussians at the midpoint and end. In refinement mode, runs a full
        densification step each iteration instead.

        Args:
            iterations: Number of optimisation iterations.
            only_frame_id: If set, always optimise against this single keyframe
                instead of sampling randomly.
            refinement: If ``True``, enables the tqdm progress bar and runs
                densification rather than mid-run pruning.
        """
        selected_frames = list(self.keyframes.keys())
        if len(selected_frames) == 0 or len(self.ignored_frames) == len(self.keyframes):
            print("no frames available")
            return

        background = torch.zeros(3).cuda()
        pipe = utils.flashsplat_pipe()
        for iteration in tqdm(range(iterations), "Refinement", disable=not refinement):
            keyframe_id = self._sample_valid_keyframe(selected_frames, only_frame_id)

            keyframe = self.keyframes[keyframe_id]
            gt_color, gt_depth = keyframe["color"].cuda().clone(), keyframe["depth"].cuda().clone()
            pose, intrinsics = keyframe["pose"].cuda().clone(), keyframe["intrinsics"]
            flashsplat_view = utils.flashsplat_cam(
                gt_color, gt_depth, None, intrinsics, pose.cpu(), keyframe_id)
            del keyframe

            render_pkg = flashsplat_render(
                flashsplat_view, self.gaussian_model, pipe, background, obj_num=self.num_label_channels)

            image, depth, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"].clone(), render_pkg["depth"].clone(),
                render_pkg["viewspace_points"], render_pkg["visibility_filter"].clone(),
                render_pkg["radii"].clone())
            viewspace_point_tensor.retain_grad()
            del render_pkg

            mask = (~torch.isnan(depth)).squeeze(0).to(image.device)
            if self.occlusion_masks.get(keyframe_id) is not None:
                mask = mask * ~self.occlusion_masks[keyframe_id].squeeze(0).to(image.device)

            color_loss = (((1.0 - self.opt_params.lambda_dssim)
                           * l1_loss(image, gt_color, agg="none") * mask).mean()
                          + (self.opt_params.lambda_dssim
                             * (1.0 - ssim(image.unsqueeze(0), gt_color.unsqueeze(0),
                                           data_range=1.,
                                           mask=mask.unsqueeze(0).tile((3, 1, 1)).unsqueeze(0)))))
            depth_loss = (l1_loss(depth, gt_depth, agg="none") * mask).mean()
            reg_loss = self.config["isotropic_reg_weight"] * isotropic_loss(self.gaussian_model.get_scaling.clone())
            total_loss = color_loss + depth_loss + reg_loss
            total_loss.backward()

            with torch.no_grad():
                if not refinement:
                    if iteration == (iterations // 2) or iteration == iterations:
                        prune_mask = (self.gaussian_model.get_opacity.detach() < 0.1).squeeze()
                        self.gaussian_model.prune_points(prune_mask)
                else:
                    self._densification_step(
                        iteration, total_loss, visibility_filter, radii, viewspace_point_tensor)
                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def _find_added_geometry_masks(self, keyframe) -> list:
        """Render the current frame and return indices of masks where new geometry appears.

        Compares rendered depth against ground-truth depth to find masks where a
        significant fraction of visible pixels are closer in reality than in the
        current model, indicating newly added geometry.

        Args:
            keyframe: Dict with keys ``color``, ``depth``, ``masks``, ``pose``,
                and ``intrinsics`` for the current frame (tensors on CUDA).

        Returns:
            List of integer mask indices that exceed the adding threshold.
        """
        EPSILON = self.config["depth_change_threshold"]
        gt_depth = keyframe["depth"]
        pose, intrinsics = keyframe["pose"], keyframe["intrinsics"]

        flashsplat_view = utils.flashsplat_cam(
            keyframe["color"], gt_depth, None, intrinsics, pose.cpu(), None)
        render_label_pkg = flashsplat_render(
            flashsplat_view, self.gaussian_model, utils.flashsplat_pipe(),
            torch.zeros(3).cuda(), gt_mask=None, obj_num=1)
        render_depth, render_alpha = render_label_pkg["depth"], render_label_pkg["alpha"]

        valid_according_2alpha = render_alpha.squeeze() > self.config["min_opacity"]
        depth_diff = (gt_depth - render_depth).squeeze()

        new_geometry_indices = []
        for i, mask in enumerate(keyframe["masks"]):
            mask_alpha = mask * valid_according_2alpha
            mask_add = mask_alpha * (depth_diff < -EPSILON)
            add_area = mask_add.sum() / (mask_alpha * (depth_diff < EPSILON)).sum()
            if add_area > self.config["addition_coverage_threshold"]:
                new_geometry_indices.append(i)
        return new_geometry_indices

    @torch.no_grad()
    def _propagate_addition_occlusions(self, keyframe, new_geometry_indices: list):
        """Reproject addition masks into covisible frames and update occlusion state.

        For each flagged mask, reprojects the corresponding depth points into every
        covisible keyframe. If the reprojected region covers a sufficient fraction of
        that frame, the area is added to ``self.occlusion_masks`` so it is excluded from
        future optimization. Frames where the added object is prominently visible are
        also added to ``self.ignored_frames``.

        Args:
            keyframe: Dict with keys ``depth``, ``masks``, ``pose``, and
                ``intrinsics`` for the current frame (tensors on CUDA).
            new_geometry_indices: List of mask indices (as returned by
                ``_find_added_geometry_masks``) that contain newly added geometry.
        """
        gt_depth = keyframe["depth"]
        pose, intrinsics = keyframe["pose"], keyframe["intrinsics"]
        covisible_ids = self.get_covisible_keyframes(keyframe)

        for i, mask in enumerate(keyframe["masks"]):
            if i not in new_geometry_indices:
                continue
            for covis_id in covisible_ids:
                cov_depth = self.keyframes[covis_id]["depth"].cuda()
                occluded, _ = utils.reproject_points(
                    gt_depth, pose, intrinsics,
                    cov_depth, self.keyframes[covis_id]["pose"].cuda(),
                    self.keyframes[covis_id]["intrinsics"], start_mask=mask)
                if not occluded.any():
                    continue

                cover_score = (((occluded > 0) * (occluded + 2.0 < cov_depth)).sum()
                               / (occluded > 0).sum())
                if cover_score > 0.01:
                    occluded = utils.np2torch(cv.morphologyEx(
                        (occluded > 0).squeeze().to(torch.uint8).cpu().numpy(),
                        cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))).cuda()
                    if covis_id not in self.occlusion_masks:
                        self.occlusion_masks[covis_id] = (occluded > 0).squeeze()
                    else:
                        self.occlusion_masks[covis_id] = (
                            self.occlusion_masks[covis_id] | (occluded > 0)).squeeze()

                if cover_score > self.config["covis_ignore_threshold"]:
                    if covis_id not in self.ignored_frames:
                        self.ignored_frames.add(covis_id)

    @torch.no_grad()
    def detect_additions(self, keyframe):
        """Detect newly added geometry in a keyframe and propagate occlusions.

        Orchestrates change detection for the adding case: first identifies which
        masks contain new geometry, then propagates the resulting occlusions into
        all covisible keyframes.

        Args:
            keyframe: Dict with keys ``color``, ``depth``, ``masks``, ``pose``,
                and ``intrinsics`` for the current frame (tensors on CUDA).
        """
        new_geometry_indices = self._find_added_geometry_masks(keyframe)
        if new_geometry_indices:
            self._propagate_addition_occlusions(keyframe, new_geometry_indices)

    @torch.no_grad()
    def _detect_conflicting_gaussians(self, keyframe):
        """Render the current frame and return a boolean mask of conflicting gaussians.

        Finds gaussians whose projected position disagrees with the observed depth
        AND color, then confirms they form a visible 2D region via a second render.

        Args:
            keyframe: Dict with keys ``color``, ``depth``, ``pose``, and
                ``intrinsics`` for the current frame (tensors on CUDA).

        Returns:
            A boolean tensor of shape ``(N,)`` masking conflicting gaussians, or
            ``None`` if no conflict is detected.
        """
        EPSILON = self.config["depth_change_threshold"]
        gt_color, gt_depth = keyframe["color"], keyframe["depth"]
        pose, intrinsics = keyframe["pose"], keyframe["intrinsics"]
        _, h, w = gt_color.shape

        flashsplat_view = utils.flashsplat_cam(
            gt_color, gt_depth, None, intrinsics, pose.cpu(), None)
        pipe = utils.flashsplat_pipe()
        render_label_pkg = flashsplat_render(
            flashsplat_view, self.gaussian_model, pipe,
            torch.zeros(3).cuda(), gt_mask=None, obj_num=1)
        render_color, render_used, render_xy, render_gs_depth = (
            render_label_pkg["render"], render_label_pkg["visibility_filter"],
            render_label_pkg["proj_xy"], render_label_pkg["gs_depth"])

        fr_render_xy = render_xy[:, render_used]
        fr_render_gs = render_gs_depth[render_used]
        inframe_filter = ((fr_render_xy[0, :] < w) * (fr_render_xy[1, :] < h) * (
            fr_render_xy[0, :] > 0) * (fr_render_xy[1, :] > 0))
        screen_xy = fr_render_xy[:, inframe_filter]
        screen_gsd = fr_render_gs[inframe_filter]

        screen_xy_int = screen_xy.to(torch.int)
        gt_depth_gs_pixel = gt_depth[screen_xy_int[1], screen_xy_int[0]]
        close_used_inframe = (gt_depth_gs_pixel - screen_gsd) > EPSILON

        if close_used_inframe.sum() == 0:
            return None

        # check for color error
        l1_color_diff = l1_loss(gt_color, render_color, agg="none").permute((1, 2, 0)).mean(-1)
        color_error_pixels = l1_color_diff[screen_xy_int[1], screen_xy_int[0]] > self.config["color_error_threshold"]
        depth_and_color_error = close_used_inframe * color_error_pixels

        if depth_and_color_error.sum() == 0:
            return None

        removal_prune_indices = torch.arange(render_used.shape[0]).cuda()
        removal_prune_indices = removal_prune_indices[render_used][inframe_filter][depth_and_color_error]
        prune_mask = torch.zeros_like(render_used).to(torch.bool).cuda()
        prune_mask[removal_prune_indices] = True

        if prune_mask.sum() == 0:
            return None

        # confirm the candidates form a visible 2D region
        conflict_gaussians_2d = flashsplat_render(
            flashsplat_view, self.gaussian_model, pipe,
            torch.ones(3).cuda(), obj_num=1, used_mask=prune_mask)
        conflict_gaussians_2d = conflict_gaussians_2d["alpha"] > self.config["min_opacity"]
        conflict_gaussians_2d = cv.erode(
            conflict_gaussians_2d.squeeze().cpu().to(torch.uint8).numpy(),
            np.ones((5, 5), np.uint8))
        conflict_gaussians_2d = utils.np2torch(conflict_gaussians_2d, "cuda")

        return prune_mask if conflict_gaussians_2d.sum() > 0 else None

    @torch.no_grad()
    def _propagate_removal_masks(self, frame_id, keyframe, removal_mask):
        """Expand removal mask using covisible frames and prune conflicting gaussians.

        For each covisible keyframe, renders only the flagged gaussians, finds masks
        that agree in depth and color (meaning those gaussians belong there), expands
        the removal set accordingly, and updates ``self.occlusion_masks``. Finally prunes
        all identified gaussians from the model.

        Args:
            frame_id: ID of the current frame (skipped in covisibility loop).
            keyframe: Dict with keys ``depth``, ``masks``, ``pose``, and
                ``intrinsics`` for the current frame (tensors on CUDA).
            removal_mask: Boolean tensor of shape ``(N,)`` identifying the initial
                set of gaussians to remove.
        """
        EPSILON = self.config["depth_change_threshold"]
        covisible_ids = self.get_covisible_keyframes(keyframe)

        any_extra_gaussian = False
        extra_removal_mask = torch.zeros_like(removal_mask).to(torch.bool)
        pipe = utils.flashsplat_pipe()
        background = torch.zeros(3).cuda()

        for covis_id in covisible_ids:
            if covis_id == frame_id:
                continue

            covisible_color_gt = self.keyframes[covis_id]["color"].cuda()
            covisible_depth_gt = self.keyframes[covis_id]["depth"].cuda()
            covisible_intrinsics = self.keyframes[covis_id]["intrinsics"]
            covisible_pose = self.keyframes[covis_id]["pose"]
            _, h, w = covisible_color_gt.shape

            covisible_flashsplat_view = utils.flashsplat_cam(
                covisible_color_gt, covisible_depth_gt, None, covisible_intrinsics, covisible_pose, covis_id)
            covisible_render_pkg = flashsplat_render(
                covisible_flashsplat_view, self.gaussian_model, pipe, background,
                gt_mask=None, used_mask=removal_mask, obj_num=1)

            covisible_render_color = covisible_render_pkg["render"]
            covisible_render_depth = covisible_render_pkg["depth"]
            covisible_render_alpha = covisible_render_pkg["alpha"]

            # where depth is +- epsilon of gt
            agree_in_depth = ((covisible_render_depth - covisible_depth_gt).abs() < EPSILON).squeeze()
            agree_in_color = (l1_loss(covisible_render_color, covisible_color_gt, agg="none"
                                      ).permute((1, 2, 0)).mean(-1) < self.config["color_error_threshold"]).squeeze()
            alpha_mask = covisible_render_alpha > self.config["min_opacity"]
            agree_depth_color_mask = agree_in_depth * agree_in_color * alpha_mask

            # add agreed areas to occlusion mask
            if covis_id not in self.occlusion_masks:
                self.occlusion_masks[covis_id] = agree_depth_color_mask.to(torch.bool).squeeze()
            else:
                self.occlusion_masks[covis_id] = (
                    self.occlusion_masks[covis_id] | agree_depth_color_mask.to(torch.bool)).squeeze()

            for mask in self.keyframes[covis_id]["masks"]:
                mask = mask.cuda()

                # if the mask is covered above threshold already, add it fully to occlusion mask
                coverage = (self.occlusion_masks[covis_id] * mask).sum() / mask.sum()
                if coverage > self.config["removal_coverage_threshold"]:
                    self.occlusion_masks[covis_id] = (
                        self.occlusion_masks[covis_id] | mask.to(torch.bool)).squeeze()

                agreement = (mask * agree_depth_color_mask).sum() / mask.sum()
                if agreement > self.config["removal_coverage_threshold"]:
                    covisible_flashsplat_view = utils.flashsplat_cam(
                        covisible_color_gt, covisible_depth_gt, mask,
                        covisible_intrinsics, covisible_pose, covis_id)
                    all_covis_render_pkg = flashsplat_render(
                        covisible_flashsplat_view, self.gaussian_model, pipe,
                        background, gt_mask=None, obj_num=self.num_label_channels)

                    covis_xy = all_covis_render_pkg["proj_xy"]
                    covis_gs_depth = all_covis_render_pkg["gs_depth"]
                    inframe_filter = (
                        covis_xy[0, :] < w) * (covis_xy[1, :] < h) * (
                        covis_xy[0, :] > 0) * (covis_xy[1, :] > 0)
                    screen_xy = covis_xy[:, inframe_filter]
                    screen_xy_int = screen_xy.to(torch.int)
                    gaussians_in_mask = mask[screen_xy_int[1], screen_xy_int[0]]
                    gaussians_fit_depth = (
                        (covis_gs_depth[inframe_filter]
                         - covisible_depth_gt[screen_xy_int[1], screen_xy_int[0]]).abs() < EPSILON)
                    gaussians_in_mask_fit_depth = (gaussians_in_mask * gaussians_fit_depth).to(torch.bool)

                    if gaussians_in_mask_fit_depth.sum() > 0:
                        any_extra_gaussian = True
                        extra_removal_mask[inframe_filter] = (
                            extra_removal_mask[inframe_filter] | gaussians_in_mask_fit_depth)
                        self.occlusion_masks[covis_id] = (
                            self.occlusion_masks[covis_id] | mask.bool()).squeeze()

        if any_extra_gaussian:
            removal_mask = removal_mask | extra_removal_mask
        self.gaussian_model.prune_points(removal_mask)

    @torch.no_grad()
    def detect_removals(self, frame_id, keyframe):
        """Detect and remove gaussians that conflict with the observed frame.

        Orchestrates the removal case: first identifies conflicting gaussians via
        depth and color disagreement, then propagates the removal across covisible
        keyframes before pruning the model.

        Args:
            frame_id: ID of the current frame.
            keyframe: Dict with keys ``color``, ``depth``, ``masks``, ``pose``,
                and ``intrinsics`` for the current frame (tensors on CUDA).
        """
        removal_mask = self._detect_conflicting_gaussians(keyframe)
        if removal_mask is not None:
            self._propagate_removal_masks(frame_id, keyframe, removal_mask)

    def is_keyframe(self, pose: np.ndarray) -> bool:
        """Decide whether a new frame should become a keyframe.

        Returns ``True`` for the very first frame. Otherwise compares the given
        pose against the last keyframe by translation distance and rotation
        magnitude in Euler angles.

        Args:
            pose: (4, 4) float32 world-to-camera pose matrix.

        Returns:
            ``True`` if the frame should be added as a keyframe.
        """
        if not self.keyframes:
            return True
        last_pose = self.keyframes[self._last_keyframe_id]["pose"].detach().numpy()
        delta_pose = np.linalg.inv(last_pose) @ pose
        translation_diff = np.linalg.norm(delta_pose[:3, 3])
        rot_euler_diff = np.abs(Rotation.from_matrix(
            (delta_pose[:3, :3])).as_euler("xyz"))
        return translation_diff > self.config["keyframe_translation_diff"] or np.any(rot_euler_diff > 50)

    def get_covisible_keyframes(self, keyframe_data):
        """Return keyframe IDs that are covisible with the given frame, sorted by overlap.

        For each stored keyframe, reprojects its depth map into the given frame
        and counts overlapping pixels as a covisibility score. Keyframes with a
        non-zero score are returned sorted by ascending score.

        Args:
            keyframe_data: Dict with keys ``depth``, ``pose``, and
                ``intrinsics`` for the query frame.

        Returns:
            List of keyframe IDs sorted by ascending covisibility score.
        """
        gt_depth, pose, intrinsics = keyframe_data["depth"], keyframe_data["pose"], keyframe_data["intrinsics"]

        covis_ids = []
        scores = []
        for covis_id in self.keyframes.keys():
            cov_depth, cov_pose, cov_intrinsics = (self.keyframes[covis_id]["depth"],
                                                   self.keyframes[covis_id]["pose"],
                                                   self.keyframes[covis_id]["intrinsics"])
            occluded_result, _ = utils.reproject_points(
                cov_depth, cov_pose, cov_intrinsics, gt_depth, pose, intrinsics)

            covis_score = (occluded_result > 0).sum()
            if covis_score > 0:
                covis_ids.append(covis_id)
                scores.append(covis_score)

        if len(scores) > 0:
            scores, covis_ids = zip(*sorted(zip(scores, covis_ids)))

        return covis_ids

    @torch.no_grad()
    def _add_gaussians(self, color, depth, segmentation, pose, intrinsics):
        """Seed new gaussians from an RGBD frame and add them to the model.

        Renders the current model to identify unoccupied or poorly-reconstructed
        regions (low alpha, depth error, or high colour loss), lifts those pixels
        to 3D, and appends the resulting gaussians via ``add_points``.

        Args:
            color: Ground-truth colour image of shape ``(3, H, W)`` on CUDA.
            depth: Ground-truth depth map of shape ``(H, W)`` on CUDA.
            segmentation: Optional segmentation mask of shape ``(H, W)`` on CUDA,
                or ``None``.
            pose: Camera-to-world pose of shape ``(4, 4)`` on CUDA.
            intrinsics: Camera intrinsic matrix of shape ``(3, 3)``.
        """
        if self.gaussian_model.get_xyz.shape[0] == 0:
            seeding_mask = torch.ones_like(depth)
        else:
            pose = pose.clone().detach().cpu()
            flashsplat_view = utils.flashsplat_cam(
                color, depth, segmentation, intrinsics, pose, None)
            pipe = utils.flashsplat_pipe()
            background = torch.ones(3).cuda()
            render_pkg = flashsplat_render(
                flashsplat_view, self.gaussian_model, pipe, background,
                obj_num=self.num_label_channels)
            rendered_depth, rendered_alpha, rendered_color = (
                render_pkg["depth"].clone(), render_pkg["alpha"].clone(),
                render_pkg["render"].clone())
            depth_error = torch.abs(depth - rendered_depth)
            depth_error_mask = (rendered_depth > depth) * (depth_error > 40 * depth_error.median())
            alpha_mask = rendered_alpha < self.config["min_opacity"]

            seeding_mask = alpha_mask | depth_error_mask
            og_seeding_img = torch.stack([seeding_mask[0]] * 3, dim=-1)
            l1_color_loss = l1_loss(rendered_color, color, agg='none').permute((1, 2, 0)).mean(-1)
            threshed_color = torch.stack(
                [(torch.ones_like(l1_color_loss) * self.config["gaussian_seed_threshold"]) < l1_color_loss
                 ] * 3, -1).bool()
            newly_added = threshed_color > og_seeding_img
            seeding_mask = newly_added[:, :, 0].bool() | seeding_mask[0]

        pose = utils.torch2np(pose)
        seeding_mask = utils.torch2np(seeding_mask).astype(np.uint8)
        color = color.clone().permute(1, 2, 0) * 255
        filtered_color = utils.torch2np(color).astype(np.uint8)
        filtered_color[seeding_mask == 0] = 0
        filtered_depth = utils.torch2np(depth.clone())
        filtered_depth[seeding_mask == 0] = 0

        cloud_to_add = utils.rgbd2ptcloud(filtered_color, filtered_depth, intrinsics, pose)
        cloud_to_add = cloud_to_add.uniform_down_sample(2)
        utils.add_points(self.gaussian_model, cloud_to_add)

    def train(self, dataset, output_path):
        """Process a dataset sequentially, adding keyframes and optimising the model.

        For each frame in the dataset, checks whether it qualifies as a keyframe.
        If so, runs change detection (detect_additions), stores the keyframe, adds new
        Gaussians, and runs two rounds of optimisation — a short single-frame pass
        followed by a multi-frame pass. Non-keyframes update only the pose log.

        Args:
            dataset: Dataset instance to train on. Must expose ``start_frame``
                and ``run_id`` attributes.
            output_path: ``Path`` under which a ``checkpoints/`` subdirectory
                is created for this run.
        """
        io_utils.setup_output_paths(output_path, ["checkpoints"])

        num_frames = len(dataset)
        for dataset_frame_idx in tqdm(range(dataset.start_frame, num_frames), f"training run {dataset.run_id}"):

            frame_id = len(self.estimated_poses)

            sample = dataset[dataset_frame_idx]
            pose, intrinsics = sample["pose"], sample["intrinsics"]

            # later will be estimated with a tracking module
            self.estimated_poses[frame_id] = pose

            if self.is_keyframe(pose):
                color, depth, masks = sample["color"], sample["depth"], sample["masks"]
                keyframe_data = {
                    "color": utils.np2torch(color, device="cuda").permute(2, 0, 1) / 255.0,
                    "depth": utils.np2torch(depth, device="cuda"),
                    "masks": masks.cuda(),
                    "pose": utils.np2torch(pose, device="cuda"),
                    "intrinsics": intrinsics,
                }
                if len(self.keyframes) > 2:
                    self.detect_additions(keyframe_data)
                self.keyframes[frame_id] = utils.dict2device(keyframe_data, "cpu")
                self._last_keyframe_id = frame_id
                self._add_gaussians(keyframe_data["color"], keyframe_data["depth"],
                                    None, keyframe_data["pose"],
                                    keyframe_data["intrinsics"])
                num_iters = self.config["first_keyframe_iters"] if frame_id == dataset.start_frame else self.config["keyframe_iters"]
                self.optimize_model(50, only_frame_id=frame_id)
                self.detect_removals(frame_id, keyframe_data)
                self.optimize_model(num_iters)

    def save(self, output_path, data_config):
        """Serialise the model and config to disk.

        Saves the Gaussian model and keyframe metadata as a checkpoint, and
        writes the full config as a JSON file alongside it.

        Args:
            output_path: Root ``Path`` under which ``checkpoints/checkpoint.pth``
                and ``configs.json`` are written.
            data_config: Data config dict stored verbatim in the checkpoint and
                the JSON config file.
        """
        model = self.gaussian_model.capture()
        io_utils.save_dict_to_ckpt({
            "keyframe_key_ids": list(self.keyframes.keys()),
            "dataset_path": data_config["dataset_path"],
            "occlusion_masks": self.occlusion_masks,
            "ignored_frames": self.ignored_frames,
            "model": model
        }, "checkpoint.pth", directory=output_path / "checkpoints")
        del model
        io_utils.save_dict_to_json(
            {"slam": self.config, "data": data_config}, directory=output_path, file_name="configs.json")
        torch.cuda.empty_cache()

    def load(self, checkpoint_path, datasets):
        """Restore model weights and keyframe state from a checkpoint.

        Replays all frames in ``datasets`` to reconstruct ``estimated_poses``
        and reloads the keyframes that were active at save time.

        Args:
            checkpoint_path: Path to the ``.pth`` checkpoint file.
            datasets: List of dataset instances in the same order used during
                the original training run. Required to rebuild keyframe tensors.
        """
        print(f"loading checkpoint from: {checkpoint_path}")
        loaded = torch.load(checkpoint_path, weights_only=False)
        model = GaussianModel(3)
        model.restore(loaded["model"], (self.opt_params))

        keyframe_ids = loaded["keyframe_key_ids"]
        self.keyframes = {}
        for run in datasets:
            for sample in run:
                frame_id = len(self.estimated_poses)
                pose = sample["pose"]
                self.estimated_poses[frame_id] = pose
                if frame_id in keyframe_ids:
                    color, depth, masks = sample["color"], sample["depth"], sample["masks"]
                    intrinsics = sample["intrinsics"]

                    keyframe_data = {
                        "color": utils.np2torch(color, device="cuda").permute(2, 0, 1) / 255.0,
                        "depth": utils.np2torch(depth, device="cuda"),
                        "masks": masks.cuda(),
                        "pose": utils.np2torch(pose, device="cuda"),
                        "intrinsics": intrinsics,
                    }
                    self.keyframes[frame_id] = utils.dict2device(
                        keyframe_data, "cpu")  # do not waste VRAM for storage
                    self._last_keyframe_id = frame_id

        self.gaussian_model = model
        self.occlusion_masks = loaded["occlusion_masks"]
        self.ignored_frames = loaded["ignored_frames"]

    def evaluate(self, dataset, output_path, split: str, refinement_state=None):
        """Render and evaluate the model against a dataset, logging metrics to wandb.

        Args:
            dataset: Dataset instance to evaluate against.
            output_path: Root output ``Path``; rendered frames are written to a
                ``<wandb-run-name>/`` (or ``local/``) subdirectory.
            split: Label for the data split, e.g. ``"train"`` or ``"test"``.
                Used in file names and as the wandb metric prefix.
            refinement_state: Optional stage label (e.g. ``"pre_ref"``,
                ``"post_ref"``) prepended to the wandb key as
                ``<refinement_state>/<split>/``.
        """
        run_dir = wandb.run.name if self.wandb_online else "local"

        eval_output_path = output_path / run_dir
        psnr, lpips, ssim, l1 = evaluate_all_rendering(
            self.gaussian_model, dataset, eval_output_path,
            refinement_state=refinement_state, set_info=f"_{split}")
        if self.wandb_online:
            prefix = f"{refinement_state}/{split}" if refinement_state else split
            wandb.log({f"{prefix}/psnr": psnr,
                       f"{prefix}/lpips": lpips,
                       f"{prefix}/ssim": ssim,
                       f"{prefix}/l1": l1})
