import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.lst_num_per_cls = np.zeros((num_classes,))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # import pdb; pdb.set_trace()
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)  # (0,1,2,3)
        return hist

    def _get_num_per_class(self, label_true):
        res = []
        for i in range(0, self.num_classes):
            res.append((label_true==i).sum())
        # import pdb; pdb.set_trace()   # [5073, 3887]
        return np.array(res)
        
    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # (112, 80)
            # import pdb; pdb.set_trace()
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            self.lst_num_per_cls += self._get_num_per_class(lt.flatten())  # (2) lt中0-1标签的数量

    def evaluate(self, flag_cls_weight=False, flag_ignore_background=False):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        if flag_cls_weight:
            if flag_ignore_background:
                tmp_weight = self.lst_num_per_cls  # [5073,3887]
                tmp_weight[0] = 0
                tmp_weight = tmp_weight / (tmp_weight.sum())   # [0,1]
                import pdb; pdb.set_trace()
            else:
                tmp_weight = self.lst_num_per_cls / (self.lst_num_per_cls.sum())
            return iu, np.nansum(iu * tmp_weight)
        else:
            return iu, np.nanmean(iu)

def obtain_meanIOU_for_each_instance_in_batch(m_pred, m_gt, num_cls, flag_weight=False, flag_ignore_bg=False):
    assert m_pred.shape == m_gt.shape
    res_miou = []
    if len(m_pred.shape) == 4:
        # import pdb; pdb.set_trace()
        for each_i in range(len(m_pred)):
            tmp_metric = meanIOU(num_cls)
            tmp_metric.add_batch(m_pred[each_i], m_gt[each_i])
            res_miou.append(tmp_metric.evaluate(flag_cls_weight=flag_weight, flag_ignore_background=flag_ignore_bg)[-1])
    else:
        raise ValueError
    # return np.array(res_miou)
    return np.array(res_miou, dtype=np.float32)


def compute_ulb_hardness_miou(current_pred,  teacher_pred, num_class,
            flag_using_cls_weighted_iou=True,
            flag_ignoring_background=True):
            
    hardness_tea = obtain_meanIOU_for_each_instance_in_batch(current_pred.cpu().numpy(), teacher_pred.cpu().numpy(), 
                num_class, flag_weight=flag_using_cls_weighted_iou, 
                flag_ignore_bg=flag_ignoring_background)
    hardness_stu = obtain_meanIOU_for_each_instance_in_batch(teacher_pred.cpu().numpy(), current_pred.cpu().numpy(), 
                num_class, flag_weight=flag_using_cls_weighted_iou, 
                flag_ignore_bg=flag_ignoring_background)
    # import pdb; pdb.set_trace()
    # if flag_ignoring_background:
    #     hardness_tea[hardness_tea<0.002] = 1.0
    #     hardness_stu[hardness_stu<0.002] = 1.0

    return hardness_tea, hardness_stu

def compute_ulb_hardness_all(stu_outputs, tea_outputs, thresh=0.95,
            flag_using_cls_weighted_iou=True,
            flag_ignoring_background=True):
    batch_size, num_class, w, h, d = stu_outputs.shape

    with torch.no_grad():
        prob_stu = torch.softmax(stu_outputs.detach(), dim=1)
        prob_tea = torch.softmax(tea_outputs.detach(), dim=1)

        max_prob_stu, pred_labl_stu = torch.max(prob_stu, dim=1)
        max_prob_tea, pred_labl_tea = torch.max(prob_tea, dim=1)

        # hardness 1: iou
        hardness_iou_tea, hardness_iou_stu = compute_ulb_hardness_miou(pred_labl_stu, pred_labl_tea, num_class, 
            flag_using_cls_weighted_iou=flag_using_cls_weighted_iou, 
            flag_ignoring_background=flag_ignoring_background)

        hardness_iou_tea = torch.from_numpy(hardness_iou_tea).cuda()    # (4)
        hardness_iou_stu = torch.from_numpy(hardness_iou_stu).cuda()
        
        # hardness 2: high-ratio
        hardness_ratio_tea = max_prob_tea.ge(thresh).float().mean(dim=[1,2,3])    # (4,80)
        hardness_ratio_stu = max_prob_stu.ge(thresh).float().mean(dim=[1,2,3])
        # import pdb; pdb.set_trace()
# =================================================================================================================================================================
        # orginal
        hardness_v1 = (hardness_iou_tea * hardness_ratio_tea + hardness_iou_stu * hardness_ratio_stu)/ 2
        # test new
        # hardness_v1 = (hardness_ratio_tea + hardness_ratio_stu)/ 2
# =================================================================================================================================================================
        # tmp_ratio = hardness_ratio_tea + hardness_ratio_stu
        # hardness_v2 = (hardness_ratio_tea / tmp_ratio) * hardness_iou_tea + (hardness_ratio_stu / tmp_ratio) * hardness_iou_stu

    # return hardness_v1, hardness_v2, hardness_ratio_stu
    return hardness_v1