import math
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.utils
import torchvision
import yaml


class BaseDataset(torch.utils.data.Dataset):
    """Base class for RGB-D datasets with SAM mask support.

    Parses camera intrinsics and optional distortion/crop parameters from a
    config dict. Subclasses are responsible for populating ``color_paths``,
    ``depth_paths``, and ``poses``.

    Args:
        dataset_config: Dict with keys ``dataset_path``, ``H``, ``W``,
            ``fx``, ``fy``, ``cx``, ``cy``, ``depth_scale``, and optionally
            ``distortion``, ``crop_edge``, ``frame_limit``.
    """

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["dataset_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.frame_ids = []
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []
        self.poses = []
        self.color_transform = torchvision.transforms.ToTensor()

    def __len__(self) -> int:
        """Returns the number of frames in the dataset."""
        return len(self.color_paths) if self.frame_limit < 0 else min(int(self.frame_limit), len(self.color_paths))

    def unpack_bitpacked_data(self, packed_bytes: np.ndarray, H: int, W: int) -> torch.Tensor:
        """Unpack a bit-packed byte array into a boolean mask tensor.

        Args:
            packed_bytes: Bit-packed array as returned by ``numpy.packbits``.
            H: Height of the original mask.
            W: Width of the original mask.

        Returns:
            Boolean tensor of shape ``(H, W)``.
        """
        unpacked_bits = np.unpackbits(packed_bytes)
        total_size = H * W

        binary_mask_np = unpacked_bits[:total_size].reshape(H, W)
        return torch.from_numpy(binary_mask_np).bool()


class AriaChangeDataset(BaseDataset):
    """RGB-D change-detection dataset recorded with an Aria device.

    Expects a directory layout of ``<dataset_path>/room<room>/<run>/`` for
    training runs and ``<dataset_path>/<run>/`` for test runs. Each run
    directory must contain ``results/frame*.jpg``, ``results/depth*.png``,
    ``traj.txt``, and ``sam_masks.h5``.

    Args:
        dataset_config: Top-level config dict whose ``data`` sub-dict is
            forwarded to ``BaseDataset``.
        room: Room index used when constructing the data path.
        run: Run subdirectory name (e.g. ``"run1"``).
        testing: If ``True``, reads directly from ``<dataset_path>/<run>``
            instead of the room-scoped path.
        n_frames: Optional cap on the number of frames returned by ``__len__``.
    """

    def __init__(self, dataset_config: dict, room, run, testing=False, n_frames=None):
        super().__init__(dataset_config["data"])
        if not testing:
            base_path = self.dataset_path / f"room{room}" / run
        else:
            base_path = self.dataset_path / run
        self.room = room
        self.n_frames = n_frames
        self.color_paths = sorted(
            list((base_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((base_path / "results").glob("depth*.png")))
        self.load_poses(base_path / "traj.txt")
        sam_h5_path = base_path / "sam_masks.h5"
        self.h5_file = h5py.File(sam_h5_path, 'r', libver='latest')
        self.frame_ids_strs = [
            f"frame{id:06d}" for id in range(len(self.color_paths))]

        # CP-SLAM dataset has a bug in terms of number of poses
        num_loaded_frames = len(self)
        self.frame_ids = list(range(num_loaded_frames))
        self.poses = self.poses[:num_loaded_frames]
        self.run_id = f"run_{run}"
        self.start_frame = 0
        print(f"Loaded {num_loaded_frames} frames")
        self.data_config = self.dataset_config

    def __len__(self):
        if self.n_frames is not None:
            return self.n_frames
        else:
            return len(self.color_paths)

    def __del__(self):
        if hasattr(self, 'h5_file') and not self.h5_file.closed:
            self.h5_file.close()

    def load_poses(self, path: str) -> None:
        """Load camera-to-world poses from a text file.

        Args:
            path: Path to a whitespace-delimited file where each line is a
                flattened 4x4 pose matrix.
        """
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def filter_data(self, frame_ids: list):
        """Restrict the dataset to a specific subset of frame indices.

        Args:
            frame_ids: List of integer indices to keep.
        """
        self.color_paths = [self.color_paths[i] for i in frame_ids]
        self.depth_paths = [self.depth_paths[i] for i in frame_ids]
        self.poses = [self.poses[i] for i in frame_ids]
        self.frame_ids_strs = [self.frame_ids_strs[i] for i in frame_ids]

    def __getitem__(self, index: int) -> dict:
        """Return a single frame as a dict.

        Args:
            index: Dataset index.

        Returns:
            Dict with keys ``frame_id`` (int), ``color`` (H, W, 3) uint8,
            ``depth`` (H, W) float32, ``masks`` bool tensor of shape
            (N, H, W), ``pose`` (4, 4) float32 world-to-camera, and
            ``intrinsics`` (3, 3) float64.
        """
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        # load sam from h5
        image_id_str = self.frame_ids_strs[index]
        current_frame_masks = []
        if image_id_str in self.h5_file:
            image_group = self.h5_file[image_id_str]
            for mask_id in sorted(image_group.keys()):
                dset = image_group[mask_id]

                packed_bytes = dset[()]
                H, W = dset.attrs["original_shape"]

                mask_tensor = self.unpack_bitpacked_data(packed_bytes, H, W)
                current_frame_masks.append(mask_tensor)

        if current_frame_masks:
            sam_masks = torch.stack(current_frame_masks)
        else:
            sam_masks = torch.empty(
                (0, self.height, self.width), dtype=torch.bool)

        return {
            "frame_id": self.frame_ids[index],
            "color": color_data,
            "depth": depth_data,
            "masks": sam_masks,
            "pose": np.linalg.inv(self.poses[index]),
            "intrinsics": self.intrinsics,
        }


def get_aria_datasets(dataset_path, room=0):
    """Build training and test dataset splits for an Aria room.

    Loads ``run1`` as the first training sequence and ``run2`` with a 9:1
    train/test split (every 10th frame held out for testing).

    Args:
        dataset_path: Root path containing ``config_GaME.yaml`` and the run
            subdirectories.
        room: Room index to load (default ``0``).

    Returns:
        Tuple of ``(training_datasets, test_datasets)`` where each element is
        a list of ``AriaChangeDataset`` instances.
    """
    dataset_config = yaml.safe_load(
        open(f"{dataset_path}/config_GaME.yaml", 'r'))

    dataset_config["data"]["dataset_path"] = dataset_path

    run1_dataset = AriaChangeDataset(
        dataset_config=dataset_config, room=room, run="run1")
    run2_dataset = AriaChangeDataset(
        dataset_config=dataset_config, room=room, run="run2")

    train_ids = [i for i in range(len(run2_dataset)) if i % 10 != 0 or i == 0]

    run2_dataset.filter_data(train_ids)
    training_datasets = [run1_dataset, run2_dataset]

    run2_dataset = AriaChangeDataset(
        dataset_config=dataset_config, room=room, run="run2")
    run2_dataset.filter_data(
        [i for i in range(len(run2_dataset)) if i not in train_ids])
    test_datasets = [run2_dataset]

    return training_datasets, test_datasets


class FlatDataset(torch.utils.data.Dataset):
    """Flat directory RGB-D dataset with per-frame pose files and SAM masks.

    Expects a run directory at ``<dataset_path>/<run_id>/`` containing files
    named ``<id>_color.png``, ``<id>_depth.tiff``, ``<id>_pose.txt``, and
    a ``sam_masks.h5`` file.

    Args:
        dataset_path: Root dataset directory.
        run_id: Subdirectory name for this run (e.g. ``"run1"``).
        data_config: Config dict with keys ``start_frame``, ``width``,
            ``height``, ``fx``, ``fy``, ``cx``, ``cy``.
    """

    def __init__(self, dataset_path: str, run_id: str, data_config: dict):
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.run_id = run_id
        self.data_config = data_config
        self.start_frame = data_config["start_frame"]

        self.width = data_config["width"]
        self.height = data_config["height"]
        self.fx, self.fy = data_config["fx"], data_config["fy"]
        self.cx, self.cy = data_config["cx"], data_config["cy"]
        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.color_transform = torchvision.transforms.ToTensor()

        run_path = self.dataset_path / run_id
        self.color_paths = sorted(list(run_path.glob("*_color.png")))
        self.depth_paths = sorted(list(run_path.glob("*_depth.tiff")))
        self.pose_paths = sorted(list(run_path.glob("*_pose.txt")))
        self.poses = self.__load_poses()
        sam_h5_path = run_path / "sam_masks.h5"
        self.h5_file = h5py.File(sam_h5_path, 'r', libver='latest')
        self.frame_ids_strs = [
            f"{id:06d}_color" for id in range(len(self.color_paths))]

    def __del__(self):
        if hasattr(self, 'h5_file') and not self.h5_file.closed:
            self.h5_file.close()

    def __load_poses(self):
        poses = []
        for pose_path in self.pose_paths:
            pose = np.loadtxt(str(pose_path))
            poses.append(pose)
        poses = np.array(poses)
        return poses

    def filter_data(self, frame_ids: list):
        """Restrict the dataset to a specific subset of frame indices.

        Args:
            frame_ids: List of integer indices to keep.
        """
        self.color_paths = [self.color_paths[i] for i in frame_ids]
        self.depth_paths = [self.depth_paths[i] for i in frame_ids]
        self.pose_paths = [self.pose_paths[i] for i in frame_ids]
        self.frame_ids_strs = [self.frame_ids_strs[i] for i in frame_ids]
        self.poses = self.__load_poses()

    def __len__(self):
        return len(self.color_paths)

    def unpack_bitpacked_data(self, packed_bytes: np.ndarray, H: int, W: int) -> torch.Tensor:
        """Unpack a bit-packed byte array into a boolean mask tensor.

        Args:
            packed_bytes: Bit-packed array as returned by ``numpy.packbits``.
            H: Height of the original mask.
            W: Width of the original mask.

        Returns:
            Boolean tensor of shape ``(H, W)``.
        """
        unpacked_bits = np.unpackbits(packed_bytes)
        total_size = H * W

        binary_mask_np = unpacked_bits[:total_size].reshape(H, W)
        return torch.from_numpy(binary_mask_np).bool()

    def __getitem__(self, idx):
        """Return a single frame as a dict.

        Args:
            idx: Dataset index.

        Returns:
            Dict with keys ``frame_id`` (int), ``color`` (H, W, 3) uint8,
            ``depth`` (H, W) float32 scaled by 10, ``masks`` bool tensor of
            shape (N, H, W), ``pose`` (4, 4) float32 world-to-camera with
            translation scaled by 10, and ``intrinsics`` (3, 3) float64.
        """
        color_data = cv2.imread(str(self.color_paths[idx]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[idx]), cv2.IMREAD_UNCHANGED)

        # load sam from h5
        image_id_str = self.frame_ids_strs[idx]
        current_frame_masks = []
        if image_id_str in self.h5_file:
            image_group = self.h5_file[image_id_str]
            for mask_id in sorted(image_group.keys()):
                dset = image_group[mask_id]

                packed_bytes = dset[()]
                H, W = dset.attrs["original_shape"]

                mask_tensor = self.unpack_bitpacked_data(packed_bytes, H, W)
                current_frame_masks.append(mask_tensor)

        if current_frame_masks:
            sam_masks = torch.stack(current_frame_masks)
        else:
            sam_masks = torch.empty(
                (0, self.height, self.width), dtype=torch.bool)

        # fix window
        window = depth_data > 10
        if window.any():
            # single pixels in image with crazy depth
            if window.sum() < 6:
                positions = np.argwhere(window)
                for pos in positions:
                    depth_data[pos[0], pos[1]] = depth_data[pos[0]+1, pos[1]]
            else:
                # this is it
                depth_data[window] = 0
                depth_data = depth_data + window * depth_data.max(0)

        pose = np.linalg.inv(self.poses[idx])  # Camera to world

        # GS rendering pipeline does not render regions with "too small depth"
        # https://github.com/graphdeco-inria/gaussian-splatting/issues/429
        # Therefore, we scale the depth values by 10
        # To accomodate this, we also scale the pose translation by 10
        pose[:3, 3] = pose[:3, 3] * 10

        frame_id = int(str(self.pose_paths[idx])[-15:-9])

        return {
            "frame_id": frame_id,
            "color": color_data,
            "depth": depth_data * 10,
            "masks": sam_masks,
            "pose": pose,
            "intrinsics": self.intrinsics,
        }


def get_flat_datasets(data_config: dict):
    """Build training and test dataset splits for a flat directory dataset.

    Loads ``run1`` fully as the first training sequence and splits ``run2``
    9:1 (every 10th frame held out for testing).

    Args:
        data_config: Config dict forwarded to ``FlatDataset``. Must contain
            ``dataset_path`` and ``start_frame``.

    Returns:
        Tuple of ``(training_datasets, test_datasets)`` where each element is
        a list of ``FlatDataset`` instances.
    """
    dataset_path = data_config["dataset_path"]
    training_datasets, test_datasets = [], []

    run1_dataset = FlatDataset(dataset_path, "run1", data_config)
    if data_config["start_frame"]:
        start_frame = data_config["start_frame"]
    else:
        start_frame = 0
    frame_ids = [i for i in range(start_frame, len(run1_dataset))]
    run1_dataset.filter_data(frame_ids)

    run2_dataset = FlatDataset(dataset_path, "run2", data_config)
    train_ids = [i for i in range(len(run2_dataset)) if i % 10 != 0 or i == 0]

    run2_dataset.filter_data(train_ids)
    training_datasets = [run1_dataset, run2_dataset]

    run2_dataset = FlatDataset(dataset_path, "run2", data_config)
    run2_dataset.filter_data(
        [i for i in range(len(run2_dataset)) if i not in train_ids])
    test_datasets = [run2_dataset]

    return training_datasets, test_datasets


class TUM_RGBDPS(BaseDataset):
    """TUM RGB-D dataset loader with pose synchronisation and SAM masks.

    Reads ``rgb.txt``, ``depth.txt``, ``groundtruth.txt`` (or ``pose.txt``),
    and ``sam_masks.h5`` from ``dataset_path``, associates frames by
    timestamp, and sub-samples at the given frame rate.

    Args:
        dataset_config: Config dict forwarded to ``BaseDataset``. Optionally
            contains ``run_id`` (defaults to ``"run1"``).
    """

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses, self.seg_maps, self.seg_labels = self.loadtum(
            self.dataset_path, frame_rate=32)
        self.start_frame = 0
        self.data_config = dataset_config
        if "run_id" in dataset_config:
            self.run_id = dataset_config["run_id"]
        else:
            self.run_id = "run1"

    def parse_list(self, filepath, skiprows=0):
        """Read a whitespace-delimited list file into a string array.

        Args:
            filepath: Path to the file.
            skiprows: Number of header rows to skip.

        Returns:
            numpy array of dtype ``str_``.
        """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.str_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """Associate image, depth, and pose entries by nearest timestamp.

        Args:
            tstamp_image: 1-D array of image timestamps.
            tstamp_depth: 1-D array of depth timestamps.
            tstamp_pose: 1-D array of pose timestamps, or ``None`` to skip
                pose association.
            max_dt: Maximum allowed time difference for a valid association.

        Returns:
            List of ``(image_idx, depth_idx)`` or
            ``(image_idx, depth_idx, pose_idx)`` tuples.
        """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """Load and sub-sample TUM RGB-D data at a target frame rate.

        Args:
            datapath: Root directory of the TUM sequence.
            frame_rate: Target frame rate for sub-sampling. ``-1`` loads all
                frames.

        Returns:
            Tuple of ``(images, depths, poses, seg_maps, seg_labels)`` where
            images and depths are lists of file paths, poses is a list of
            (4, 4) float32 arrays in a coordinate frame relative to the first
            frame, and seg_maps / seg_labels are empty lists reserved for
            future use.
        """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')
        sam_h5_path = os.path.join(datapath, 'sam_masks.h5')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        self.h5_file = h5py.File(sam_h5_path, 'r', libver='latest')
        self.frame_ids_strs = [p[0] for p in self.parse_list(image_list)]

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        seg_maps, seg_labels = [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]

            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses, seg_maps, seg_labels

    def pose_matrix_from_quaternion(self, pvec):
        """Convert a translation + quaternion vector to a 4x4 pose matrix.

        Args:
            pvec: Array of length 7: ``[tx, ty, tz, qx, qy, qz, qw]``.

        Returns:
            (4, 4) float64 pose matrix.
        """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __del__(self):
        if hasattr(self, 'h5_file') and not self.h5_file.closed:
            self.h5_file.close()

    def __getitem__(self, index):
        """Return a single frame as a dict.

        Args:
            index: Dataset index.

        Returns:
            Dict with keys ``frame_id`` (int), ``color`` (H, W, 3) uint8,
            ``depth`` (H, W) float32 with missing values inpainted,
            ``masks`` bool tensor of shape (N, H, W), ``pose`` (4, 4) float32
            world-to-camera, and ``intrinsics`` (3, 3) float64.
        """
        color_data = cv2.imread(str(self.color_paths[index]))

        # load sam from h5
        image_id_str = self.frame_ids_strs[index]
        current_frame_masks = []
        if image_id_str in self.h5_file:
            image_group = self.h5_file[image_id_str]
            for mask_id in sorted(image_group.keys()):
                dset = image_group[mask_id]

                packed_bytes = dset[()]
                H, W = dset.attrs["original_shape"]

                mask_tensor = self.unpack_bitpacked_data(packed_bytes, H, W)
                current_frame_masks.append(mask_tensor)

        if current_frame_masks:
            sam_masks = torch.stack(current_frame_masks)
        else:
            sam_masks = torch.empty((0, H, W), dtype=torch.bool)

        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            sam_masks = sam_masks[:, edge:-edge, edge:-edge]

        # interpolate depth for missing values
        fill_depth = cv2.inpaint(depth_data, (depth_data == 0).astype(
            np.uint8), cv2.INPAINT_TELEA, None)

        return {
            "frame_id": index,
            "color": color_data,
            "depth": fill_depth,
            "masks": sam_masks,
            "pose": np.linalg.inv(self.poses[index]),
            "intrinsics": self.intrinsics,
        }


def get_datasets(data_config: dict):
    """Instantiate training and test datasets from a data config dict.

    Dispatches to the appropriate dataset loader based on ``dataset_name``.
    Add new dataset types here.

    Args:
        data_config: Dict containing at least ``dataset_name``. Additional
            keys are forwarded to the underlying loader.

    Returns:
        Tuple of ``(training_datasets, test_datasets)``, each a list of
        dataset instances.

    Raises:
        ValueError: If ``dataset_name`` is not specified.
        NotImplementedError: If ``dataset_name`` is not recognised.
    """
    dataset_name = data_config.get("dataset_name")
    if dataset_name is None:
        raise ValueError("Please specify dataset_name in the config file")
    if dataset_name == "flat_dataset":
        return get_flat_datasets(data_config)
    if dataset_name in ("aria_0", "aria_1"):
        return get_aria_datasets(data_config["dataset_path"], data_config["aria_room"])
    if dataset_name in ("tum_desk", "tum_xyz", "tum_office"):
        return [TUM_RGBDPS(data_config)], []
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
