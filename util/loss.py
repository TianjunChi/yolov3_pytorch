import torch
import torch as T
import math
import numpy as np
from util.decode import decode 

def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:

    举例时假设pred_xywh和label_xywh的shape都是(1, 4)
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = T.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2_x0y0x1y1 = T.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = T.cat((T.min(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
                             T.max(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])), dim=-1)
    boxes2_x0y0x1y1 = T.cat((T.min(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
                             T.max(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])), dim=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
                boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
    boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
                boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = T.max(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    right_down = T.min(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = right_down - left_up
    inter_section = T.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = T.min(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
    enclose_right_down = T.max(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = T.pow(enclose_wh[..., 0], 2) + T.pow(enclose_wh[..., 1], 2)

    # 两矩形中心点距离的平方
    p2 = T.pow(boxes1[..., 0] - boxes2[..., 0], 2) + T.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。加上除0保护防止nan。
    atan1 = T.atan(boxes1[..., 2] / (boxes1[..., 3] + 1e-9))
    atan2 = T.atan(boxes2[..., 2] / (boxes2[..., 3] + 1e-9))
    v = 4.0 * T.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou

def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 150, 4)
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = T.cat((boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                    boxes1[..., :2] + boxes1[..., 2:] * 0.5), dim=-1)
    boxes2 = T.cat((boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                    boxes2[..., :2] + boxes2[..., 2:] * 0.5), dim=-1)

    # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 150, 2)
    left_up = T.max(boxes1[..., :2], boxes2[..., :2])  # 相交矩形的左上角坐标
    right_down = T.min(boxes1[..., 2:], boxes2[..., 2:])  # 相交矩形的右下角坐标

    # 相交矩形的w和h，是负数时取0     (?, grid_h, grid_w, 3, 150, 2)
    inter_section = right_down - left_up
    inter_section = T.where(inter_section < 0.0, inter_section*0, inter_section)
    inter_area = inter_section[..., 0] * inter_section[..., 1]  # 相交矩形的面积            (?, grid_h, grid_w, 3, 150)
    union_area = boxes1_area + boxes2_area - inter_area  # union_area      (?, grid_h, grid_w, 3, 150)
    iou = 1.0 * inter_area / union_area  # iou                             (?, grid_h, grid_w, 3, 150)
    return iou

def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh, alpha=0.5, gamma=2):
    conv_shape = conv.shape
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_prob = pred[:, :, :, :, 5:]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = bbox_ciou(pred_xywh, label_xywh)                             # (8, 13, 13, 3)
    ciou = ciou.reshape((batch_size, output_size, output_size, 3, 1))   # (8, 13, 13, 3, 1)
    input_size = float(input_size)

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_pos_loss = label_prob * (0 - T.log(pred_prob + 1e-9))             # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_neg_loss = (1 - label_prob) * (0 - T.log(1 - pred_prob + 1e-9))   # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_mask = respond_bbox.repeat((1, 1, 1, 1, num_class))
    prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个focal_loss
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算focal_loss。
    expand_pred_xywh = pred_xywh[:, :, :, :, np.newaxis, :]  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :]  # 扩展为(?,      1,      1, 1, 150, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  150个ground truth  计算iou。   (?, grid_h, grid_w, 3, 150)
    max_iou, max_iou_indices = T.max(iou, dim=-1, keepdim=True)        # 与150个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共150个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多150个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * (max_iou < iou_loss_thresh).float()

    # focal_loss介绍： https://www.cnblogs.com/king-lps/p/9497836.html  公式简单，但是效果出群！alpha解决不平衡问题，gamma解决困难样本问题。
    # 为什么正样本数量少，给的权重alpha比负样本的权重(1-alpha)还小？ 请看 https://blog.csdn.net/weixin_44638957/article/details/100733971

    # YunYang1994的focal_loss，只带gamma解决困难样本问题。没有带上alpha。
    # pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9)) * T.pow(1 - pred_conf, gamma)
    # neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9)) * T.pow(pred_conf, gamma)

    # RetinaNet的focal_loss，多带上alpha解决不平衡问题。
    # 经过试验发现alpha取>0.5的值时mAP会提高，但误判（False Predictions）会增加；alpha取<0.5的值时mAP会降低，误判会降低。
    # pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9)) * T.pow(1 - pred_conf, gamma) * alpha
    # neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9)) * T.pow(pred_conf, gamma) * (1 - alpha)

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - T.log(pred_conf + 1e-9))
    neg_loss = respond_bgd  * (0 - T.log(1 - pred_conf + 1e-9))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算，其实对yolov3算法是有好处的。（论文里称之为ignore）
    # 它如果作为反例参与置信度loss的计算，会降低yolov3的精度。
    # 它如果作为正例参与置信度loss的计算，可能会导致预测的框不准确（因为可能物体的中心都预测不准）。

    ciou_loss = ciou_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的ciou_loss，再求平均值
    conf_loss = conf_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的conf_loss，再求平均值
    prob_loss = prob_loss.sum((1, 2, 3, 4)).mean()    # 每个样本单独计算自己的prob_loss，再求平均值
    print('ciou loss: ',ciou_loss,'conf loss: ', conf_loss, 'prob loss: ',prob_loss)
    return ciou_loss + conf_loss + prob_loss 

def yolo_loss(args, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3):
    conv_lbbox = args[0]   # (?, ?, ?, 3, num_classes+5)
    conv_mbbox = args[1]   # (?, ?, ?, 3, num_classes+5)
    conv_sbbox = args[2]   # (?, ?, ?, 3, num_classes+5)
    label_sbbox = args[3]   # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]   # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]   # (?, ?, ?, 3, num_classes+5)
    true_sbboxes = args[6]   # (?, 150, 4)
    true_mbboxes = args[7]   # (?, 150, 4)
    true_lbboxes = args[8]   # (?, 150, 4)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32)
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbboxes, 8, num_classes, iou_loss_thresh, alpha=alpha_1)
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbboxes, 16, num_classes, iou_loss_thresh, alpha=alpha_2)
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbboxes, 32, num_classes, iou_loss_thresh, alpha=alpha_3)
    print('sbbox: ',loss_sbbox,' mbbox: ',loss_mbbox,' lbbox: ',loss_lbbox)
    return loss_sbbox + loss_mbbox + loss_lbbox

class YoloLoss(torch.nn.Module):
    def __init__(self, num_classes, iou_loss_thresh, anchors, alpha_1, alpha_2, alpha_3):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss_thresh = iou_loss_thresh
        self.anchors = anchors
        self.alpha_1 = alpha_1    # 小感受野输出层的focal_loss的alpha
        self.alpha_2 = alpha_2    # 中感受野输出层的focal_loss的alpha
        self.alpha_3 = alpha_3    # 大感受野输出层的focal_loss的alpha

    def forward(self, args):
        return yolo_loss(args, self.num_classes, self.iou_loss_thresh, self.anchors, self.alpha_1, self.alpha_2, self.alpha_3)
