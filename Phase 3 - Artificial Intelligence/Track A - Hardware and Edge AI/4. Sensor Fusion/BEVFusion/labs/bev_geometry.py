"""CPU-tested PyTorch geometry kernels, not a pretrained BEVFusion detector."""
import torch


def lift_pixels(pixels, depth, intrinsics, camera_to_ego_rotation, camera_to_ego_translation, augmentation):
    """B,N,P pixels; metric optical-axis depth; affine homogeneous image augmentation.

    pixels: B,N,P,2. depth: B,N,P. Matrices: B,N,3,3. Translation: B,N,3.
    augmentation maps ORIGINAL image homogeneous pixels to AUGMENTED pixels.
    Inputs must already use a compatible pinhole/undistorted image model.
    """
    if pixels.ndim != 4 or pixels.shape[-1] != 2 or depth.shape != pixels.shape[:-1]:
        raise ValueError('expected pixels B,N,P,2 and depths B,N,P')
    if not torch.isfinite(pixels).all() or not torch.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('finite pixels and positive metric depth required')
    homogeneous = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    original = torch.linalg.solve(augmentation, homogeneous.transpose(-1, -2)).transpose(-1, -2)
    original = original / original[..., 2:3]
    rays = torch.linalg.solve(intrinsics, original.transpose(-1, -2)).transpose(-1, -2)
    camera = rays * depth[..., None]  # preserve metric depth after undoing pixel augmentation
    return (camera_to_ego_rotation @ camera.transpose(-1, -2)).transpose(-1, -2) + camera_to_ego_translation[..., None, :]


def bev_sum(points, features, xbounds, ybounds, zbounds):
    """Sum B,N,P,C features into B,C,Y,X, preserving batch and filtering height.

    x/y bounds=(minimum, maximum exclusive, cell size). z bounds=(min,max exclusive).
    Height bins are collapsed; this is a small reference, not the optimized CUDA pool.
    """
    if points.ndim != 4 or points.shape[-1] != 3 or points.shape[:-1] != features.shape[:-1]:
        raise ValueError('points B,N,P,3 and features B,N,P,C required')
    if not torch.isfinite(points).all() or not torch.isfinite(features).all():
        raise ValueError('finite features and points required')
    sizes = []
    for low, high, step in (xbounds, ybounds):
        if step <= 0 or high <= low or abs((high-low)/step-round((high-low)/step)) > 1e-8:
            raise ValueError('positive cell size and integer number of cells required')
        sizes.append(round((high-low)/step))
    nx, ny = sizes
    b, n, p, c = features.shape
    ix = torch.floor((points[..., 0]-xbounds[0])/xbounds[2]).long()
    iy = torch.floor((points[..., 1]-ybounds[0])/ybounds[2]).long()
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (points[..., 2] >= zbounds[0]) & (points[..., 2] < zbounds[1])
    batch = torch.arange(b, device=points.device)[:, None, None].expand(b, n, p)
    flat_index = (batch * ny + iy) * nx + ix
    result = features.new_zeros((b*ny*nx, c))
    result.index_add_(0, flat_index[valid], features[valid])
    return result.view(b, ny, nx, c).permute(0, 3, 1, 2).contiguous()
