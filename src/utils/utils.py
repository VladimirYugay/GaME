import os
import random
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from simple_knn._C import distCUDA2

from src.flashsplat.scene.cameras import Camera
from src.flashsplat.utils.general_utils import inverse_sigmoid
from src.flashsplat.utils.sh_utils import RGB2SH


def flashsplat_pipe():
    """Build a minimal pipeline config object expected by FlashSplat renderers.

    Returns:
        A namespace object with ``compute_cov3D_python``, ``convert_SHs_python``,
        and ``debug`` attributes set to ``False``.
    """
    class GroupParams:
        pass
    fake_pipeline = GroupParams()
    setattr(fake_pipeline, 'compute_cov3D_python', False)
    setattr(fake_pipeline, 'convert_SHs_python', False)
    setattr(fake_pipeline, 'debug', False)

    return fake_pipeline


def flashsplat_cam(gt_color, gt_depth, segmentation, intrinsics, pose, uid):
    """Construct a FlashSplat Camera from standard RGBD frame data.

    Args:
        gt_color: Ground-truth color image tensor of shape ``(3, H, W)``.
        gt_depth: Ground-truth depth map tensor of shape ``(H, W)``.
        segmentation: Optional segmentation mask tensor of shape ``(H, W)``,
            or ``None``.
        intrinsics: Camera intrinsic matrix of shape ``(3, 3)``.
        pose: Camera-to-world pose matrix of shape ``(4, 4)`` on CPU.
        uid: Unique identifier for this camera view.

    Returns:
        A ``Camera`` instance ready for use with ``flashsplat_render``.
    """
    camera = Camera(colmap_id=None,
                    R=torch2np(torch.inverse(pose))[:3, :3],
                    T=pose[:3, 3],
                    FoVx=2*np.arctan(intrinsics[0, 2] / intrinsics[0, 0]),
                    FoVy=2*np.arctan(intrinsics[1, 2] / intrinsics[1, 1]),
                    image=gt_color,
                    gt_alpha_mask=None,
                    image_name=None,
                    uid=uid,
                    gt_depth=gt_depth,
                    objects=segmentation,
                    c2w=pose)
    return camera


def setup_seed(seed: int) -> None:
    """Set the seed for reproducibility across torch, numpy, and random.

    Args:
        seed: Seed value applied to all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy ndarray.

    Args:
        tensor: The PyTorch tensor to convert.

    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.clone().detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """Convert a numpy array of points to an Open3D point cloud.

    Args:
        pts: Point coordinates of shape ``(N, 3)``.
        rgb: Optional colour values of shape ``(N, 3)`` in ``[0, 1]``.

    Returns:
        An ``o3d.geometry.PointCloud`` with points (and colours if provided).
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    """Send all tensors in a dictionary to a specified device.

    Non-tensor values are deep-copied unchanged.

    Args:
        dict: Dictionary whose tensor values should be moved.
        device: Target device string. Defaults to ``'cpu'``.

    Returns:
        A new dictionary with all tensors moved to ``device``.
    """
    new_dict = {}
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.clone().to(device)
        else:
            new_dict[k] = deepcopy(v)
    return new_dict


def rgbd2ptcloud(img, depth, intrinsics, pose=np.eye(4)):
    """Convert an RGBD image to an Open3D point cloud.

    Args:
        img: RGB image of shape ``(H, W, 3)`` with values in ``[0, 255]``.
        depth: Depth map of shape ``(H, W)`` in metres.
        intrinsics: Camera intrinsic matrix of shape ``(3, 3)``.
        pose: 4×4 extrinsic matrix to transform points into world space.
            Defaults to identity.

    Returns:
        An ``o3d.geometry.PointCloud`` in world coordinates.
    """
    height, width, _ = img.shape
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(img)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        convert_rgb_to_intensity=False,
        depth_scale=1.0,
        depth_trunc=100,
    )
    intrinsics = o3d.open3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=intrinsics[0][0],
        fy=intrinsics[1][1],
        cx=intrinsics[0][2],
        cy=intrinsics[1][2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_img, intrinsics, extrinsic=pose, project_valid_depth_only=True)


def reproject_points(start_depth, start_pose, start_intrinsics, end_depth,
                     end_pose, end_intrinsics, start_mask=None):
    """Reproject depth points from one camera frame into another.

    Lifts pixels from the start frame into 3D using ``start_depth`` and
    ``start_pose``, then projects them into the end frame. Returns two depth
    maps in the end frame's image space: one with occlusion filtering and one
    without.

    Args:
        start_depth: Depth map of the source frame, shape ``(H, W)``.
        start_pose: World-to-camera pose of the source frame, shape ``(4, 4)``.
        start_intrinsics: Intrinsic matrix of the source camera, shape ``(3, 3)``.
        end_depth: Depth map of the target frame, shape ``(H, W)``.
        end_pose: World-to-camera pose of the target frame, shape ``(4, 4)``.
        end_intrinsics: Intrinsic matrix of the target camera, shape ``(3, 3)``.
        start_mask: Optional mask of shape ``(H, W)`` to restrict which source
            pixels are lifted. Defaults to ``None`` (all pixels).

    Returns:
        A tuple ``(occluded_result, img)`` where both are depth maps of shape
        ``(H, W)`` on CUDA. ``occluded_result`` contains only points whose
        reprojected depth is within 1 unit of the target frame's observed depth;
        ``img`` contains all reprojected depths without occlusion filtering.
    """
    cov_depth, cov_pose, cov_intrinsics = start_depth.cuda(
    ), start_pose.cuda(), start_intrinsics
    gt_depth, pose, intrinsics = end_depth.cuda(), end_pose.cuda(), end_intrinsics
    h, w = cov_depth.shape[0], cov_depth.shape[1]
    fx, fy, cx, cy = cov_intrinsics[0, 0], cov_intrinsics[1,
                                                          1], cov_intrinsics[0, 2], cov_intrinsics[1, 2]
    o3d_intrinsics = o3d.cuda.pybind.camera.PinholeCameraIntrinsic(
        w, h, fx, fy, cx, cy)

    if start_mask is not None:
        o3d_image = o3d.geometry.Image((cov_depth * start_mask).cpu().numpy())
    else:
        o3d_image = o3d.geometry.Image(cov_depth.cpu().numpy())

    o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_image, o3d_intrinsics, np.eye(4), project_valid_depth_only=True)
    points_no_pose = np2torch(np.asarray(o3d_pc.points)).cuda()

    # transformation fun
    points_no_pose_hom = torch.cat([points_no_pose, torch.ones(
        points_no_pose.shape[0]).unsqueeze(-1).cuda()], 1)
    world_points = (torch.linalg.inv(cov_pose) @ points_no_pose_hom.T).T
    kf_space = pose @ world_points.T
    depth_per_point = kf_space.T[:, 2]
    image_points_hom = np2torch(intrinsics).cuda() @ kf_space[:3, :]
    # take out points behind camera
    infront_of_cam = image_points_hom[2] > 0
    image_points_hom = image_points_hom[:, infront_of_cam]
    depth_per_point = depth_per_point[infront_of_cam]

    image_points = image_points_hom[:2, :] / image_points_hom[2]
    rounded_points = image_points.to(int)
    valid_points_mask = (
        rounded_points[0] < w) * (rounded_points[1] < h) * (rounded_points > 0).all(0)
    rounded_points = rounded_points[:, valid_points_mask]
    depth_per_point = depth_per_point[valid_points_mask]
    x = rounded_points[0].long()
    y = rounded_points[1].long()
    d = depth_per_point
    img = torch.zeros((h, w)).cuda()
    img[y, x] = d
    # occlude depth
    gt_depth_per_point = gt_depth[y, x]
    depth_valid_points = (d - 1 < gt_depth_per_point)
    x = x[depth_valid_points]
    y = y[depth_valid_points]
    d = d[depth_valid_points]
    occluded_result = torch.zeros((h, w)).cuda()
    occluded_result[y, x] = d

    return occluded_result, img


def add_points(gaussian_model, pcd: o3d.geometry.PointCloud):
    """Initialise new gaussians from a point cloud and add them to the model.

    Converts the point cloud into gaussian parameters (position, colour as
    spherical harmonics, scale derived from nearest-neighbour distances,
    identity rotation, and 0.5 opacity), then calls
    ``densification_postfix`` to append them to the model.

    Args:
        gaussian_model: The ``GaussianModel`` to extend (on CUDA).
        pcd: Open3D point cloud with ``.points`` and ``.colors`` populated.
    """
    size = gaussian_model.get_xyz.shape[0]
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(
        np.asarray(pcd.colors)).float().cuda())
    features = (torch.zeros(
        (fused_color.shape[0], 3, (gaussian_model.max_sh_degree + 1) ** 2)).float().cuda())
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    global_points = torch.cat((gaussian_model.get_xyz.cuda(
    ), torch.from_numpy(np.asarray(pcd.points)).float().cuda()))
    dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001) * 0.05
    dist2 = dist2[size:]
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1
    opacities = inverse_sigmoid(
        0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(
        1, 2).contiguous().requires_grad_(True))
    new_features_rest = nn.Parameter(features[:, :, 1:].transpose(
        1, 2).contiguous().requires_grad_(True))
    new_scaling = nn.Parameter(scales.requires_grad_(True))
    new_rotation = nn.Parameter(rots.requires_grad_(True))
    new_opacities = nn.Parameter(opacities.requires_grad_(True))
    gaussian_model.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    )
