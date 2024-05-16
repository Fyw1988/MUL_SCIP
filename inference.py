import os
import argparse
import torch

from networks.net_factory_3d import net_factory_3d
from networks.GFN import GFN
from utils.test_patch import test_all_case
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--model_name', type=str, default='GFMT', help='model_name')
parser.add_argument('--model_type', type=str, default='vnet_trans', help='model_type')
parser.add_argument('--save_result', type=str, default=True, help='save result or not')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./model/{}".format(FLAGS.dataset_name)
test_save_path = "./model/{}/{}_predictions/".format(FLAGS.dataset_name, FLAGS.model_name)

num_classes = 2
if FLAGS.dataset_name == "LA":
    FLAGS.patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'datasets/LA'
    image_list = glob(FLAGS.root_path + '/Testing Set/*/mri_norm2.h5')
elif FLAGS.dataset_name == "Pancreas":
    FLAGS.patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'datasets/Pancreas-CT'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
elif FLAGS.dataset_name == "BraTS":
    FLAGS.patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'datasets/BraTS'
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]

if FLAGS.save_result:
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)


def test_calculate_metric():
    net = net_factory_3d(net_type=FLAGS.model_type, in_chns=1, class_num=num_classes, mode="test")
    net_gf = GFN().cuda()
    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_name))
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model_name))
    save_mode_path_gf = os.path.join(snapshot_path, '{}_best_model_gf.pth'.format(FLAGS.model_name))
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    net_gf.load_state_dict(torch.load(save_mode_path_gf), strict=False)

    print("init weight from {}".format(save_mode_path))
    net.eval()
    net_gf.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                                   patch_size=FLAGS.patch_size, stride_xy=18, stride_z=4, lgf=net_gf,
                                   save_result=FLAGS.save_result, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                                   patch_size=FLAGS.patch_size, stride_xy=16, stride_z=16, lgf=net_gf,
                                   save_result=FLAGS.save_result, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "BraTS":
        avg_metric = test_all_case(FLAGS.model_name, 1, net, image_list, num_classes=num_classes,
                                   patch_size=FLAGS.patch_size, stride_xy=64, stride_z=64, lgf=net_gf,
                                   save_result=FLAGS.save_result, test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail, nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)