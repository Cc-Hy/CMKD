from skimage import io
import torch
import numpy as np
import cv2
import kornia
import mmcv

def featuremap_to_greymap(feature_map):
    """
    feature_map: (C, sizey, sizex)
    grey_map: (sizey, sizex)
    """
    import torch
    import numpy as np
    import cv2
    if len(feature_map.shape) == 3:
        feature_map = feature_map.unsqueeze(dim=0) # (b, c, sizey, sizex)
    elif len(feature_map.shape) == 4:
        pass
    else:
        raise NotImplementedError 
    # 1. GPA, (B, C, sizey, sizex) -> (B, C, 1, 1)
    channel_weights = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1,1))
    # 2. reweighting sum cross channels, (B, C, sizey, sizex) -> (B, sizey, sizex) -> (sizey, sizex)
    reduced_map = (channel_weights * feature_map).sum(dim=1).squeeze(dim=0)
    # 3. clamp
    reduced_map = torch.relu(reduced_map)
    # 4. normalize
    a_min = torch.min(reduced_map)
    a_max = torch.max(reduced_map)
    normed_map = (reduced_map - a_min) / (a_max - a_min)
    # 5. output
    grey_map = normed_map
    return grey_map


def greymap_to_rgbimg(map_grey, background=None, background_ratio=0.2, CHW_format=False):
    """
    map_grey: np, (sizey, sizex), values in 0-1
    background: np, (sizey, sizex, 3), values in 0-255.
    """
    import torch
    import numpy as np
    import cv2
    if background is None:
        background = np.zeros((map_grey.shape[0], map_grey.shape[1], 3))
    map_uint8 = (255 * map_grey).astype(np.uint8) # 0-255
    map_bgr = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET) # 0-255
    map_rbg = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
    map_img = map_rbg + background_ratio * background
    map_img = np.clip(map_img, a_min=0, a_max=255).astype(np.uint8)
    if CHW_format:
        # (sizey, sizex, 3) -> (3, sizey, sizex)
        map_img = np.transpose(map_img, (2,0,1))
    return map_img


def draw_bev(x):     # x:tensor(C,H,W)
    # gray = featuremap_to_greymap(x.permute(0,2,1).flip(dims=[1]))

    gray = featuremap_to_greymap(x)

    rgb = greymap_to_rgbimg(gray.cpu().detach().numpy())

    import matplotlib.pyplot as plt

    plt.imshow(rgb)


def draw_box_on_img(boxes, img, extrinsic, intrinsic):

    """
    boxes: torch.tensor (N,7)
    img
    extrinsic: torch.tensor (4,4)
    intrinsic: torch.tensor (3,4)
    """
    boxes = boxes.detach()
    corners = boxes_to_corners_3d(boxes)
    intrinsic = intrinsic.unsqueeze(0)
    extrinsic = extrinsic.unsqueeze(0)
    intrinsic = kornia.geometry.conversions.convert_affinematrix_to_homography3d(intrinsic)
    trans_l2i = torch.matmul(intrinsic,extrinsic)
    img_corners = kornia.geometry.linalg.transform_points(trans_l2i, corners)
    img_corners = kornia.geometry.conversions.convert_points_from_homogeneous(img_corners)
    img_corners = img_corners.cpu().numpy().astype(np.int)
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),(4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    for corners in img_corners:
        for start, end in line_indices:
                    cv2.line(img, (corners[start, 0], corners[start, 1]),
                            (corners[end, 0], corners[end, 1]), (0,255,0), 1 ,cv2.LINE_AA)
    
    mmcv.imshow((img))


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d
