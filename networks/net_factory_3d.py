from networks.vnet import VNet
from networks.vnet_pst import VNetPST
from networks.vnet_trans import TransVNet


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, mode='train'):
    if net_type == "vnet" and mode == 'train':
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet" and mode == 'test':
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet_pst" and mode == 'train':
        net = VNetPST(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_pst" and mode == 'test':
        net = VNetPST(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet_trans" and mode == 'train':
        net = TransVNet(n_channels=in_chns, n_classes=class_num,
                    normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "vnet_trans" and mode == 'test':
        net = TransVNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=False).cuda()

    elif net_type == "vnet_trans_md" and mode == 'train':
        net = TransVNetMD(n_channels=in_chns, n_classes=class_num,
                    normalization='batchnorm', has_dropout=True).cuda()

    return net


# def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, mode='train'):
#     if net_type == "vnet" and mode == 'train':
#         net = VNet(n_channels=in_chns, n_classes=class_num,
#                    normalization='batchnorm', has_dropout=True)
#     elif net_type == "vnet" and mode == 'test':
#         net = VNet(n_channels=in_chns, n_classes=class_num,
#                    normalization='batchnorm', has_dropout=False)
#
#     elif net_type == "vnet_pst" and mode == 'train':
#         net = VNetPST(n_channels=in_chns, n_classes=class_num,
#                       normalization='batchnorm', has_dropout=True)
#     elif net_type == "vnet_pst" and mode == 'test':
#         net = VNetPST(n_channels=in_chns, n_classes=class_num,
#                       normalization='batchnorm', has_dropout=False)
#
#     elif net_type == "vnet_trans" and mode == 'train':
#         net = TransVNet(n_channels=in_chns, n_classes=class_num,
#                         normalization='batchnorm', has_dropout=True)
#     elif net_type == "vnet_trans" and mode == 'test':
#         net = TransVNet(n_channels=in_chns, n_classes=class_num,
#                         normalization='batchnorm', has_dropout=False)
#
#     elif net_type == "vnet_trans_md" and mode == 'train':
#         net = TransVNetMD(n_channels=in_chns, n_classes=class_num,
#                           normalization='batchnorm', has_dropout=True)
#
#     return net