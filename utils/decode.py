import torch
import torch as T
import math
import numpy as np

def get_grid_offset(grid_n):
    grid_offset = np.arange(grid_n)
    grid_x_offset = np.tile(grid_offset, (grid_n, 1))
    grid_y_offset = np.copy(grid_x_offset)
    grid_y_offset = grid_y_offset.transpose(1, 0)
    grid_x_offset = np.reshape(grid_x_offset, (grid_n, grid_n, 1, 1))
    grid_x_offset = np.tile(grid_x_offset, (1, 1, 3, 1))
    grid_y_offset = np.reshape(grid_y_offset, (grid_n, grid_n, 1, 1))
    grid_y_offset = np.tile(grid_y_offset, (1, 1, 3, 1))
    grid_offset = np.concatenate([grid_x_offset, grid_y_offset], axis=-1)
    return grid_offset

def decode(conv_output, anchors, stride):
    conv_shape = conv_output.shape
    output_size = conv_shape[1]

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5: ]

    grid_offset = get_grid_offset(output_size)

    # pytorch支持张量Tensor和标量相加（相乘），而不支持张量Tensor和同shape或不同shape的ndarray相加（相乘）。
    # pytorch支持张量Tensor和同shape或不同shape的Tensor相加（相乘）。
    grid_offset = torch.Tensor(grid_offset.astype(np.float32))
    anchor_t = torch.Tensor(np.copy(anchors).astype(np.float32))
    if T.cuda.is_available():
        grid_offset = grid_offset.cuda()
        anchor_t = anchor_t.cuda()

    # T.sigmoid(conv_raw_dxdy)的shape是(N, n, n, 3, 2)，grid_offset的shape是(n, n, 3, 2)。属于不同shape相加
    pred_xy = (T.sigmoid(conv_raw_dxdy) + grid_offset) * stride
    pred_wh = (T.exp(conv_raw_dwdh) * anchor_t) * stride
    pred_xywh = T.cat((pred_xy, pred_wh), dim=-1)

    pred_conf = T.sigmoid(conv_raw_conf)
    pred_prob = T.sigmoid(conv_raw_prob)
    return T.cat((pred_xywh, pred_conf, pred_prob), dim=-1)