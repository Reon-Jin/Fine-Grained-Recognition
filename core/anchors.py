import numpy as np
from config import INPUT_SIZE

_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48,
         scale=[2 ** (1. / 3.), 2 ** (2. / 3.)],
         aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96,
         scale=[2 ** (1. / 3.), 2 ** (2. / 3.)],
         aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192,
         scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)],
         aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=INPUT_SIZE):
    """
    generate default anchor

    :param anchors_setting: list of dicts, each with keys:
        - 'layer': name
        - 'stride': feature map stride
        - 'size': base size
        - 'scale': list of scales
        - 'aspect_ratio': list of aspect ratios
    :param input_shape: tuple (H, W) of input image size
    :return:
      center_anchors: (N, 4) array of (center_y, center_x, h, w)
      edge_anchors:   (N, 4) array of (y0, x0, y1, x1)
      anchor_areas:   (N,)  array of h*w
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        # feature-map size for this level
        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(int)  # <= 修复点
        H, W = output_map_shape
        ostart = stride / 2.0

        # grid of center positions
        ys = np.arange(ostart, ostart + stride * H, stride, dtype=np.float32)
        xs = np.arange(ostart, ostart + stride * W, stride, dtype=np.float32)
        oy = ys.reshape(H, 1)
        ox = xs.reshape(1, W)

        # base template
        template = np.zeros((H, W, 4), dtype=np.float32)
        template[..., 0] = oy  # center_y
        template[..., 1] = ox  # center_x

        for scale in scales:
            for ar in aspect_ratios:
                ctr = template.copy()
                ctr[..., 2] = size * scale / (ar ** 0.5)  # height
                ctr[..., 3] = size * scale * (ar ** 0.5)  # width

                # convert to edge coords
                half = ctr[..., 2:4] / 2.0
                edge = np.concatenate((
                    ctr[..., :2] - half,  # y0, x0
                    ctr[..., :2] + half   # y1, x1
                ), axis=-1)

                area = ctr[..., 2] * ctr[..., 3]

                center_anchors = np.vstack((center_anchors, ctr.reshape(-1, 4)))
                edge_anchors = np.vstack((edge_anchors, edge.reshape(-1, 4)))
                anchor_areas = np.hstack((anchor_areas, area.reshape(-1)))

    return center_anchors, edge_anchors, anchor_areas


def hard_nms(cdds, topn=10, iou_thresh=0.25):
    """
    :param cdds: (N, >=5) ndarray, each row [score, y0, x0, y1, x1, ...]
    :param topn: number of boxes to keep
    :param iou_thresh: IoU threshold for suppression
    :return: (M, 5) kept boxes
    """
    if not (isinstance(cdds, np.ndarray) and cdds.ndim == 2 and cdds.shape[1] >= 5):
        raise TypeError('cdds should be an N×5+ ndarray')

    boxes = cdds.copy()
    # sort by score ascending, so highest are at end
    order = np.argsort(boxes[:, 0])
    boxes = boxes[order]

    keep = []
    while boxes.shape[0] > 0 and len(keep) < topn:
        # pick highest remaining
        box = boxes[-1]
        keep.append(box)
        boxes = boxes[:-1]

        if boxes.shape[0] == 0:
            break

        # compute IoU with the selected box
        y0 = np.maximum(boxes[:, 1], box[1])
        x0 = np.maximum(boxes[:, 2], box[2])
        y1 = np.minimum(boxes[:, 3], box[3])
        x1 = np.minimum(boxes[:, 4], box[4])

        inter_h = np.clip(y1 - y0, a_min=0, a_max=None)
        inter_w = np.clip(x1 - x0, a_min=0, a_max=None)
        inter = inter_h * inter_w

        area_boxes = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        area_box = (box[3] - box[1]) * (box[4] - box[2])
        union = area_boxes + area_box - inter

        iou = inter / union
        # keep those with IoU < threshold
        boxes = boxes[iou < iou_thresh]

    return np.array(keep)


if __name__ == '__main__':
    # 简单测试 hard_nms
    test_boxes = np.array([
        [0.4, 1, 10, 12, 20],
        [0.5, 1, 11, 11, 20],
        [0.55, 20, 30, 40, 50]
    ], dtype=np.float32)
    kept = hard_nms(test_boxes, topn=100, iou_thresh=0.4)
    print(kept)
