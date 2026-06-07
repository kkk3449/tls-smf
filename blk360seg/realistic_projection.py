"""Realistic Projection from PointCLIP V2 (depth-map multi-view renderer).

Vendored from PointCLIP_V2/zeroshot_cls/trainers/mv_utils_zs.py with one change:
the single `torch_scatter.scatter(..., reduce="max")` call is replaced by native
`Tensor.scatter_reduce_(reduce="amax")`, so torch_scatter (no cu128 wheel) is not
needed. Renders a point cloud into `num_views` smoothed depth images for CLIP.
"""
import numpy as np
import torch
import torch.nn as nn

TRANS = -1.5

params = {'maxpoolz': 1, 'maxpoolxy': 7, 'maxpoolpadz': 0, 'maxpoolpadxy': 2,
          'convz': 1, 'convxy': 3, 'convsigmaxy': 3, 'convsigmaz': 1,
          'convpadz': 0, 'convpadxy': 1,
          'imgbias': 0., 'depth_bias': 0.2, 'obj_ratio': 0.8, 'bg_clr': 0.0,
          'resolution': 112, 'depth': 8}


def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    return kernel / kernel.sum()


def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth // 2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None, :, :], depth, axis=0) * zkernel[:, None, None]
    return kernel3d / torch.sum(kernel3d)


class Grid2Image(nn.Module):
    """3D occupancy grid -> 2D depth image: maxpool (densify), gaussian conv
    (smooth), max over depth (squeeze)."""
    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False
        self.maxpool = nn.MaxPool3d(
            (params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']),
            stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'],
                               params['maxpoolpadxy']))
        self.conv = nn.Conv3d(
            1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
            stride=1, padding=(params['convpadz'], params['convpadxy'], params['convpadxy']),
            bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'],
                                   sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1, 1, 1, 1, 1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.maxpool(x.unsqueeze(1))
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        img = 1 - img
        return img.repeat(1, 3, 1, 1)


def euler2mat(angle):
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim, _view = 0, [3, 3]
    else:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim, _view = 1, [b, 3, 3]
    cosz, sinz = torch.cos(z), torch.sin(z)
    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero, sinz, cosz, zero, zero, zero, one], dim=_dim).reshape(_view)
    cosy, siny = torch.cos(y), torch.sin(y)
    ymat = torch.stack([cosy, zero, siny, zero, one, zero, -siny, zero, cosy], dim=_dim).reshape(_view)
    cosx, sinx = torch.cos(x), torch.sin(x)
    xmat = torch.stack([one, zero, zero, zero, cosx, -sinx, zero, sinx, cosx], dim=_dim).reshape(_view)
    return xmat @ ymat @ zmat


def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    """Quantize [B,N,3] points to a [B,depth,res,res] occupancy/height grid."""
    batch, pnum, _ = points.shape
    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = ((pmax + pmin) / 2)[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']

    depth_bias = params['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)
    _x.ceil_(); _y.ceil_(); z_int = _z.ceil()
    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    coordinates = (z_int * resolution * resolution + _y * resolution + _x).long()
    grid = torch.ones([batch, depth, resolution, resolution],
                      device=points.device).view(batch, -1) * params['bg_clr']
    # native replacement for torch_scatter.scatter(reduce="max")
    grid.scatter_reduce_(1, coordinates, _z, reduce="amax", include_self=True)
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0, 1, 3, 2))
    return grid


class RealisticProjection:
    """Render [B,N,3] points into B*num_views depth images (num_views=10)."""
    def __init__(self, device="cuda"):
        self.device = device
        _views = np.asarray([
            [[1 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[5 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[7 * np.pi / 4, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[1 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[2 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[3 * np.pi / 2, 0, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, -np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
            [[0, np.pi / 2, np.pi / 2], [-0.5, -0.5, TRANS]],
        ])
        _views_bias = np.asarray([
            [[0, np.pi / 9, 0], [-0.5, 0, TRANS]]] * 8 +
            [[[0, np.pi / 15, 0], [-0.5, 0, TRANS]]] * 2)
        self.num_views = _views.shape[0]
        angle = torch.tensor(_views[:, 0, :]).float().to(device)
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        angle2 = torch.tensor(_views_bias[:, 0, :]).float().to(device)
        self.rot_mat2 = euler2mat(angle2).transpose(1, 2)
        self.translation = torch.tensor(_views[:, 1, :]).float().to(device).unsqueeze(1)
        self.grid2image = Grid2Image().to(device)

    def get_img(self, points):
        b = points.shape[0]
        v = self.translation.shape[0]
        _points = self.point_transform(
            points=torch.repeat_interleave(points, v, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1),
            rot_mat2=self.rot_mat2.repeat(b, 1, 1),
            translation=self.translation.repeat(b, 1, 1))
        grid = points2grid(_points, params['resolution'], params['depth']).squeeze(0)
        return self.grid2image(grid)

    @staticmethod
    def point_transform(points, rot_mat, rot_mat2, translation):
        rot_mat = rot_mat.to(points.device)
        rot_mat2 = rot_mat2.to(points.device)
        translation = translation.to(points.device)
        points = torch.matmul(points, rot_mat)
        points = torch.matmul(points, rot_mat2)
        return points - translation
