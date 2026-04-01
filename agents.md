# GaME

A research mapping system that combines 3D Gaussian Splatting with explicit change detection. The model (GaME) builds a Gaussian scene representation from an RGB-D stream, detects geometry additions and removals between runs, and propagates changes across covisible keyframes.

---

## Entry Point

```
python run.py --config_path <config.yaml> --output_path <dir> --data_path <dir>
```

Pipeline: train → evaluate → refine → evaluate → save

---

## Directory Structure

```
change_slam/
├── run.py                              # Main entry point
├── configs/                            # YAML configs per dataset
│   ├── aria/{room0,room1}.yaml
│   ├── flat/flat.yaml
│   └── tum/{desk,office,xyz}.yaml
└── src/
    ├── entities/
    │   ├── game.py                     # GaME: core SLAM model
    │   ├── datasets.py                 # Dataset loaders (Aria, Flat, TUM)
    │   ├── arguments.py                # OptimizationParams for Gaussian training
    │   └── losses.py                   # l1_loss, isotropic_loss
    ├── utils/
    │   ├── utils.py                    # Geometry, tensor conversion, Gaussian seeding
    │   ├── io_utils.py                 # YAML/JSON/checkpoint I/O, directory setup
    │   └── mapping_eval.py             # PSNR/LPIPS/SSIM/L1 evaluation + video export
    └── flashsplat/                     # Gaussian splatting rendering (adapted 3DGS)
        ├── gaussian_renderer/__init__.py  # flashsplat_render, render
        ├── scene/
        │   ├── gaussian_model.py       # GaussianModel: params, optimizer, densification
        │   └── cameras.py              # Camera, MiniCam
        └── utils/
            ├── general_utils.py        # inverse_sigmoid, build_rotation, lr scheduling
            ├── sh_utils.py             # RGB2SH, SH2RGB, eval_sh, SH constants
            ├── graphics_utils.py       # getWorld2View2, getProjectionMatrix, fov2focal
            └── system_utils.py         # mkdir_p, searchForMaxIteration
```

---

## Core Module: `src/entities/game.py`

**Class `GaME`** — the entire SLAM system.

### Key Attributes
| Attribute | Type | Description |
|---|---|---|
| `keyframes` | `dict[int, dict]` | Stored keyframes: color, depth, masks, pose, intrinsics (kept on CPU) |
| `gaussian_model` | `GaussianModel` | The live 3DGS scene representation |
| `occlusion_masks` | `dict[int, Tensor]` | Per-keyframe bool masks of occluded regions; excluded from optimization loss |
| `ignored_frames` | `set[int]` | Keyframe IDs fully excluded from optimization sampling |
| `estimated_poses` | `dict[int, ndarray]` | Pose per frame (identity tracking for now) |
| `num_objects` | `int` | Number of object labels (from `config["obj_num"]`) |

### Method Reference
| Method | Description |
|---|---|
| `train(dataset, output_path)` | Main loop: keyframe selection, change detection, Gaussian seeding, optimization |
| `optimize_model(iterations, only_frame_id, refinement)` | Gradient optimization; refinement=True enables densification |
| `detect_additions(keyframe)` | Detect newly added geometry; propagate occlusions to covisible frames |
| `detect_removals(frame_id, keyframe)` | Detect conflicting Gaussians; prune them and update occlusion masks |
| `_find_added_geometry_masks(keyframe)` | Returns list of mask indices where GT depth < rendered depth by epsilon |
| `_propagate_addition_occlusions(keyframe, new_geometry_indices)` | Reproject addition masks into covisible keyframes; update `occlusion_masks` and `ignored_frames` |
| `_detect_conflicting_gaussians(keyframe)` | Returns bool Gaussian mask where depth AND color disagree with observation |
| `_propagate_removal_masks(frame_id, keyframe, removal_mask)` | Expand removal to covisible frames; prune Gaussians from model |
| `is_keyframe(pose)` | True if translation or rotation delta exceeds config threshold |
| `get_covisible_keyframes(keyframe_data)` | Returns keyframe IDs sorted by depth reprojection overlap |
| `_sample_valid_keyframe(selected_frames, only_frame_id)` | Randomly sample a non-ignored, non-occluded keyframe |
| `_densification_step(...)` | Densify-and-prune + opacity reset during refinement |
| `evaluate(dataset, output_path, split, refinement_state)` | Render all frames, compute PSNR/LPIPS/SSIM/L1, log to wandb |
| `save(output_path, data_config)` | Save checkpoint + configs.json |
| `load(checkpoint_path, datasets)` | Restore model and keyframe state |

### Keyframe Dict Schema
```python
{
    "color":      Tensor (3, H, W),   # float, [0, 1]
    "depth":      Tensor (H, W),      # float, metres
    "masks":      Tensor (N, H, W),   # bool, SAM instance masks
    "pose":       Tensor (4, 4),      # camera-to-world
    "intrinsics": ndarray (3, 3),     # K matrix
}
```

---

## Rendering: `src/flashsplat/gaussian_renderer/__init__.py`

**`flashsplat_render(viewpoint_cam, gaussian_model, pipe, background, ...)`**

Extended renderer used throughout GaME. Returns:
```python
{
    "render":             Tensor (3, H, W),   # rendered color
    "depth":              Tensor (1, H, W),   # rendered depth
    "alpha":              Tensor (1, H, W),   # rendered opacity
    "visibility_filter":  Tensor (N,) bool,   # which Gaussians are visible
    "radii":              Tensor (N,),        # screen-space radii
    "viewspace_points":   Tensor,             # for grad accumulation
    "proj_xy":            Tensor (2, N),      # 2D projections per Gaussian
    "gs_depth":           Tensor (N,),        # depth per Gaussian
}
```

**`render(...)`** — standard rasterizer variant (used for eval, no mask support).

---

## Gaussian Model: `src/flashsplat/scene/gaussian_model.py`

**Class `GaussianModel`**

Stores Gaussian parameters as `nn.Parameter` tensors, all on CUDA.

| Property | Shape | Description |
|---|---|---|
| `get_xyz` | (N, 3) | 3D positions |
| `get_scaling` | (N, 3) | Scales (exp-activated) |
| `get_rotation` | (N, 4) | Quaternions (normalized) |
| `get_opacity` | (N, 1) | Opacity (sigmoid-activated) |
| `get_features` | (N, 3, (D+1)²) | Spherical harmonic coefficients |

Key methods: `training_setup`, `densification_postfix`, `prune_points`, `densify_and_prune`, `reset_opacity`, `capture` / `restore`.

---

## Utilities: `src/utils/utils.py`

| Function | Description |
|---|---|
| `flashsplat_pipe()` | Creates dummy pipeline config object expected by renderer |
| `flashsplat_cam(gt_color, gt_depth, segmentation, intrinsics, pose, uid)` | Constructs `Camera` from RGBD tensors |
| `torch2np(tensor)` | Clone + detach + cpu + numpy |
| `np2torch(array, device)` | numpy → float32 torch tensor on device |
| `dict2device(dict, device)` | Move all tensors in dict to device (deep-copy non-tensors) |
| `np2ptcloud(pts, rgb)` | numpy → Open3D PointCloud |
| `rgbd2ptcloud(img, depth, intrinsics, pose)` | RGBD image → Open3D PointCloud in world space |
| `reproject_points(start_depth, start_pose, start_intrinsics, end_depth, end_pose, end_intrinsics, start_mask)` | Reproject depth from one camera into another; returns (occluded, unfiltered) depth maps |
| `add_gaussians(gaussian_model, color, depth, segmentation, pose, intrinsics, ...)` | Render, find poorly-reconstructed pixels, seed new Gaussians |
| `add_points(gaussian_model, pcd)` | Initialize and append Gaussians from Open3D point cloud |

`inverse_sigmoid` and `RGB2SH` are imported from `src.flashsplat.utils`.

---

## Datasets: `src/entities/datasets.py`

**`get_datasets(data_config)`** — returns `(train_datasets, test_datasets)` lists.

Supported `dataset_name` values: `"flat_dataset"`, `"aria_0"`, `"aria_1"`, `"tum_desk"`, `"tum_xyz"`, `"tum_office"`.

Each dataset's `__getitem__` returns:
```python
{
    "color":      ndarray (H, W, 3),    # uint8
    "depth":      ndarray (H, W),       # float32, metres
    "masks":      Tensor (N, H, W),     # bool
    "pose":       ndarray (4, 4),       # camera-to-world, float32
    "intrinsics": ndarray (3, 3),       # K matrix, float64
}
```

Masks are stored bit-packed in HDF5 and decompressed via `unpack_bitpacked_data()`.

---

## Config Schema (YAML)

```yaml
project_name: str
wandb: bool
seed: int
output_path: str

slam:
  # Iteration counts
  first_iter: int         # Iterations for the very first keyframe
  iters: int              # Iterations per subsequent keyframe
  refinement_iters: int   # Iterations for final global refinement

  # Change detection thresholds
  epsilon: float          # Depth difference to flag conflict (metres)
  e_opacity: float        # Alpha below which a region is considered empty
  L1_thresh: float        # Color L1 error threshold for conflict detection
  adding_thresh: float    # Fraction of mask pixels that must show addition
  removal_thresh: float   # Fraction of mask agreement to trigger removal propagation
  frame_ignore_thresh: float  # Mask coverage to fully ignore a frame
  add_cover_score: float  # Cover score to add frame to ignored_frames

  # Gaussian seeding
  color_adding_thresh: float  # L1 color loss above which pixels are re-seeded

  # Keyframe selection
  keyframe_translation_diff: float  # Translation delta (metres)

  # Optimization
  reg_weight: float           # Isotropic regularization weight
  obj_num: int                # Number of object labels for renderer

  # Densification schedule
  opacity_reset_interval: int
  densify_from_iter: int
  densification_interval: int
  densify_grad_threshold: float
  min_grad: float
  scale: float                # World scale multiplier

data:
  dataset_name: str
  dataset_path: str
  start_frame: int
  run_id: int
  # Intrinsics (flat/aria): fx, fy, cx, cy, width, height
  # Intrinsics (TUM):       fx, fy, cx, cy, H, W, depth_scale, crop_edge
```

---

## Key External Dependencies

| Package | Role |
|---|---|
| `torch` | Core ML framework |
| `open3d` | 3D geometry (point clouds, reprojection) |
| `simple_knn` | KNN distance for Gaussian scale init |
| `diff-gaussian-rasterization` | Standard Gaussian rasterizer (submodule) |
| `flashsplat-rasterization` | Extended rasterizer with mask/label support (submodule) |
| `pytorch-msssim` | DSSIM loss component |
| `torchmetrics` | LPIPS metric |
| `scipy` | Rotation matrix → Euler conversion |
| `h5py` | Reading bit-packed SAM masks |
| `wandb` | Experiment tracking |
| `cv2` | Morphological ops on masks |

---

## Change Detection Algorithm Summary

**Additions** (`detect_additions`):
1. Render current model at new keyframe pose.
2. For each SAM mask: if GT depth is significantly closer than rendered depth over enough pixels → flag as addition.
3. Reproject flagged mask depth into all covisible keyframes.
4. Where reprojected region covers the covisible frame: add to `occlusion_masks`; if heavily covered, add to `ignored_frames`.

**Removals** (`detect_removals`):
1. Render current model. Find Gaussians whose projected 3D depth is behind GT depth by epsilon AND whose color disagrees.
2. Re-render only flagged Gaussians; confirm they form a visible 2D region (erosion check).
3. For each covisible frame: where flagged Gaussians agree in depth+color with that frame, expand removal set and update `occlusion_masks`.
4. Prune all flagged Gaussians from `gaussian_model`.
