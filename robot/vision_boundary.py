"""Single-view RGB-D boundary partition baseline utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage


def _normalize01(arr: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    out = np.array(arr, dtype=np.float32, copy=True)
    if valid_mask is None:
        valid_mask = np.isfinite(out)
    if not np.any(valid_mask):
        return np.zeros_like(out, dtype=np.float32)
    vals = out[valid_mask]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if hi - lo < 1e-8:
        result = np.zeros_like(out, dtype=np.float32)
        result[valid_mask] = 0.0
        return result
    out = (out - lo) / (hi - lo)
    out[~valid_mask] = 0.0
    return np.clip(out, 0.0, 1.0)


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32) / 255.0
    return (
        0.299 * rgb_f[..., 0]
        + 0.587 * rgb_f[..., 1]
        + 0.114 * rgb_f[..., 2]
    )


def _compute_rgb_edge_map(rgb: np.ndarray) -> np.ndarray:
    gray = _grayscale(rgb)
    gray = ndimage.gaussian_filter(gray, sigma=1.0)
    gx = ndimage.sobel(gray, axis=1, mode="nearest")
    gy = ndimage.sobel(gray, axis=0, mode="nearest")
    mag = np.hypot(gx, gy)
    return _normalize01(mag)


def _backproject_depth(depth: np.ndarray, intrinsics: Dict[str, float]) -> np.ndarray:
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    z = depth.astype(np.float32)
    x = ((uu - cx) / fx) * z
    y = ((vv - cy) / fy) * z
    points = np.stack([x, y, z], axis=-1)
    invalid = ~np.isfinite(z) | (z <= 1e-6)
    points[invalid] = np.nan
    return points


def _compute_depth_discontinuity_map(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    safe_depth = np.where(valid_mask, depth, np.nan)
    depth_log = np.log(np.clip(safe_depth, 1e-5, None))
    gx = np.nan_to_num(ndimage.sobel(depth_log, axis=1, mode="nearest"), nan=0.0)
    gy = np.nan_to_num(ndimage.sobel(depth_log, axis=0, mode="nearest"), nan=0.0)
    mag = np.hypot(gx, gy)
    return _normalize01(mag, valid_mask=valid_mask)


def _compute_normals_from_points(points_cam: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    # Central differences in camera frame.
    px = np.roll(points_cam, -1, axis=1) - np.roll(points_cam, 1, axis=1)
    py = np.roll(points_cam, -1, axis=0) - np.roll(points_cam, 1, axis=0)
    n = np.cross(px, py)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    n[~valid_mask] = np.nan
    n[~np.isfinite(n).all(axis=-1)] = np.nan
    return n.astype(np.float32)


def _compute_normal_discontinuity_map(normals_cam: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    h, w, _ = normals_cam.shape
    out = np.zeros((h, w), dtype=np.float32)

    right = np.roll(normals_cam, -1, axis=1)
    down = np.roll(normals_cam, -1, axis=0)

    valid_right = valid_mask & np.roll(valid_mask, -1, axis=1)
    valid_down = valid_mask & np.roll(valid_mask, -1, axis=0)

    dot_r = np.sum(normals_cam * right, axis=-1)
    dot_d = np.sum(normals_cam * down, axis=-1)
    dot_r = np.clip(dot_r, -1.0, 1.0)
    dot_d = np.clip(dot_d, -1.0, 1.0)
    ang_r = np.arccos(dot_r) / np.pi
    ang_d = np.arccos(dot_d) / np.pi

    out = np.maximum(
        np.where(valid_right, ang_r, 0.0),
        np.where(valid_down, ang_d, 0.0),
    ).astype(np.float32)
    out[~valid_mask] = 0.0
    return _normalize01(out, valid_mask=valid_mask)


def _transform_points_to_world(points_cam: np.ndarray, extrinsics: Dict[str, Any]) -> np.ndarray:
    t_world_from_camera = np.array(extrinsics["world_from_camera"], dtype=np.float64)
    rot = t_world_from_camera[:3, :3]
    trans = t_world_from_camera[:3, 3]
    pts = points_cam.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    out = np.full_like(pts, np.nan, dtype=np.float64)
    out[valid] = (rot @ pts[valid].T).T + trans
    return out.reshape(points_cam.shape).astype(np.float32)


def _transform_normals_to_world(normals_cam: np.ndarray, extrinsics: Dict[str, Any]) -> np.ndarray:
    t_world_from_camera = np.array(extrinsics["world_from_camera"], dtype=np.float64)
    rot = t_world_from_camera[:3, :3]
    n = normals_cam.reshape(-1, 3)
    valid = np.isfinite(n).all(axis=1)
    out = np.full_like(n, np.nan, dtype=np.float64)
    out[valid] = n[valid] @ rot.T
    return out.reshape(normals_cam.shape).astype(np.float32)


def _compute_support_surface_mask(
    points_world: np.ndarray,
    normals_world: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(valid_mask):
        return np.zeros_like(depth, dtype=bool)

    up_align = np.abs(normals_world[..., 2])
    up_align = np.where(np.isfinite(up_align), up_align, 0.0)

    z_vals = points_world[..., 2][valid_mask]
    d_vals = depth[valid_mask]
    z_low = float(np.quantile(z_vals, 0.40))
    d_far = float(np.quantile(d_vals, 0.72))

    horizontal = (up_align >= np.cos(np.deg2rad(35.0))) & (points_world[..., 2] <= (z_low + 0.12))
    vertical_far = (up_align <= 0.28) & (depth >= d_far)

    support = valid_mask & (horizontal | vertical_far)
    return support


def _fuse_boundary_score(
    rgb_edge: np.ndarray,
    depth_disc: np.ndarray,
    normal_disc: np.ndarray,
    support_mask: np.ndarray,
    valid_mask: np.ndarray,
    rgb_weight: float = 0.32,
    depth_weight: float = 0.42,
    normal_weight: float = 0.26,
    support_transition_weight: float = 0.16,
) -> np.ndarray:
    score = (
        rgb_weight * rgb_edge
        + depth_weight * depth_disc
        + normal_weight * normal_disc
    )
    support_boundary = ndimage.binary_dilation(support_mask, iterations=1) ^ ndimage.binary_erosion(
        support_mask, iterations=1
    )
    score = score + support_transition_weight * support_boundary.astype(np.float32)
    score[~valid_mask] = 0.0
    return np.clip(score, 0.0, 1.0)


def _relabel_compact(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    for new_id, old_id in enumerate(unique_ids, start=1):
        out[labels == old_id] = new_id
    return out


def _partition_regions(
    boundary_score: np.ndarray,
    valid_mask: np.ndarray,
    min_region_pixels: int = 45,
    boundary_quantile: float = 0.82,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.zeros_like(boundary_score, dtype=np.int32)
    if not np.any(valid_mask):
        return labels, np.zeros_like(boundary_score, dtype=bool)

    q = float(np.clip(boundary_quantile, 0.55, 0.98))
    threshold = float(np.quantile(boundary_score[valid_mask], q))
    boundary_mask = valid_mask & (boundary_score >= threshold)
    boundary_mask = ndimage.binary_dilation(boundary_mask, iterations=1)

    region_seed = valid_mask & (~boundary_mask)
    labeled, _ = ndimage.label(region_seed)
    if labeled.max() == 0:
        return labels, boundary_mask

    counts = np.bincount(labeled.ravel())
    small_ids = np.where(counts < int(min_region_pixels))[0]
    small_ids = small_ids[small_ids > 0]
    if small_ids.size > 0:
        small_mask = np.isin(labeled, small_ids)
        labeled[small_mask] = 0

    labels = _relabel_compact(labeled)
    return labels, boundary_mask


def _compute_region_stats(
    labels: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    support_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    max_id = int(labels.max())
    for region_id in range(1, max_id + 1):
        mask = labels == region_id
        count = int(mask.sum())
        if count == 0:
            continue
        ys, xs = np.where(mask)
        mean_depth = float(np.nanmean(depth[mask]))
        mean_color = rgb[mask].mean(axis=0)
        support_ratio = float(np.mean(support_mask[mask]))
        stats.append(
            {
                "region_id": int(region_id),
                "pixel_count": count,
                "centroid_uv": [float(np.mean(xs)), float(np.mean(ys))],
                "bbox_uv": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "mean_depth": mean_depth,
                "mean_color_rgb": [float(mean_color[0]), float(mean_color[1]), float(mean_color[2])],
                "support_ratio": support_ratio,
            }
        )
    stats.sort(key=lambda x: x["pixel_count"], reverse=True)
    return stats


def _gray_to_rgb_u8(gray01: np.ndarray) -> np.ndarray:
    g = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _labels_to_color_u8(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if labels.max() <= 0:
        return out
    # Deterministic pseudo-random palette.
    for rid in np.unique(labels):
        if rid <= 0:
            continue
        color = np.array(
            [
                (37 * int(rid) + 53) % 256,
                (83 * int(rid) + 97) % 256,
                (151 * int(rid) + 29) % 256,
            ],
            dtype=np.uint8,
        )
        out[labels == rid] = color
    return out


def _blend_overlay(base_rgb: np.ndarray, mask: np.ndarray, color_rgb: Tuple[int, int, int], alpha: float) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    color = np.array(color_rgb, dtype=np.float32)
    idx = mask.astype(bool)
    out[idx] = (1.0 - alpha) * out[idx] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_debug_visualizations(
    rgb: np.ndarray,
    rgb_edge: np.ndarray,
    depth_disc: np.ndarray,
    normal_disc: np.ndarray,
    support_mask: np.ndarray,
    boundary_score: np.ndarray,
    boundary_mask: np.ndarray,
    region_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    rgb_edge_vis = _gray_to_rgb_u8(rgb_edge)
    depth_disc_vis = _gray_to_rgb_u8(depth_disc)
    normal_disc_vis = _gray_to_rgb_u8(normal_disc)
    boundary_score_vis = _gray_to_rgb_u8(boundary_score)
    support_mask_vis = _blend_overlay(np.zeros_like(rgb), support_mask, (70, 150, 255), 0.85)
    region_labels_vis = _labels_to_color_u8(region_labels)
    boundary_overlay = _blend_overlay(rgb, boundary_mask, (255, 35, 35), 0.85)
    support_overlay = _blend_overlay(rgb, support_mask, (70, 180, 255), 0.55)

    top = np.concatenate([rgb, boundary_overlay, region_labels_vis, support_overlay], axis=1)
    bottom = np.concatenate([rgb_edge_vis, depth_disc_vis, normal_disc_vis, boundary_score_vis], axis=1)
    mosaic = np.concatenate([top, bottom], axis=0)

    return {
        "rgb_edge_vis": rgb_edge_vis,
        "depth_discontinuity_vis": depth_disc_vis,
        "normal_discontinuity_vis": normal_disc_vis,
        "support_surface_vis": support_mask_vis,
        "boundary_score_vis": boundary_score_vis,
        "boundary_overlay_vis": boundary_overlay,
        "region_labels_vis": region_labels_vis,
        "debug_mosaic_vis": mosaic,
    }


def compute_boundary_partition(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: Dict[str, Any],
    extrinsics: Dict[str, Any],
    *,
    min_region_pixels: int = 45,
    boundary_quantile: float = 0.82,
) -> Dict[str, Any]:
    """Compute single-view RGB-D boundary partition baseline outputs."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape [H, W, 3]")
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if rgb.shape[:2] != depth.shape:
        raise ValueError("rgb/depth spatial shape mismatch")

    h, w = depth.shape
    valid_mask = np.isfinite(depth) & (depth > 1e-6)

    rgb_edge = _compute_rgb_edge_map(rgb)
    depth_disc = _compute_depth_discontinuity_map(depth, valid_mask)

    points_cam = _backproject_depth(depth, intrinsics)
    normals_cam = _compute_normals_from_points(points_cam, valid_mask)
    normal_disc = _compute_normal_discontinuity_map(normals_cam, valid_mask)

    points_world = _transform_points_to_world(points_cam, extrinsics)
    normals_world = _transform_normals_to_world(normals_cam, extrinsics)
    support_mask = _compute_support_surface_mask(points_world, normals_world, depth, valid_mask)

    boundary_score = _fuse_boundary_score(rgb_edge, depth_disc, normal_disc, support_mask, valid_mask)
    region_labels, boundary_mask = _partition_regions(
        boundary_score,
        valid_mask,
        min_region_pixels=min_region_pixels,
        boundary_quantile=boundary_quantile,
    )

    region_stats = _compute_region_stats(region_labels, rgb, depth, support_mask)
    visuals = _build_debug_visualizations(
        rgb=rgb,
        rgb_edge=rgb_edge,
        depth_disc=depth_disc,
        normal_disc=normal_disc,
        support_mask=support_mask,
        boundary_score=boundary_score,
        boundary_mask=boundary_mask,
        region_labels=region_labels,
    )

    return {
        "width": int(w),
        "height": int(h),
        "valid_pixel_count": int(valid_mask.sum()),
        "rgb_edge_map": rgb_edge.astype(np.float32),
        "depth_discontinuity_map": depth_disc.astype(np.float32),
        "normal_discontinuity_map": normal_disc.astype(np.float32),
        "support_surface_mask": support_mask.astype(np.uint8),
        "boundary_score_map": boundary_score.astype(np.float32),
        "boundary_mask": boundary_mask.astype(np.uint8),
        "region_labels": region_labels.astype(np.int32),
        "region_stats": region_stats,
        "visualizations": visuals,
        "parameters": {
            "min_region_pixels": int(min_region_pixels),
            "boundary_quantile": float(boundary_quantile),
            "fusion_weights": {
                "rgb_edge": 0.32,
                "depth_discontinuity": 0.42,
                "normal_discontinuity": 0.26,
                "support_transition_boost": 0.16,
            },
            "support_heuristic": {
                "horizontal_normal_alignment_min": float(np.cos(np.deg2rad(35.0))),
                "vertical_normal_alignment_max": 0.28,
                "z_low_quantile": 0.40,
                "depth_far_quantile": 0.72,
            },
        },
    }
