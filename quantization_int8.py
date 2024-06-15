import torch
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
from pytorch_quantization import quant_modules
import argparse
from configs.segmentation import set_cfg_from_file
import numpy as np
from models.segmentation import model_factory
import os
from torch.utils.data import DataLoader
from lib.segmentation.data.syt_segm_dataset import SytSegmDataset
from tqdm import tqdm


quant_modules.initialize()
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

parse = argparse.ArgumentParser() 
parse.add_argument('--config', dest='config', type=str, default='configs/segmentation/bisenetv2_syt_segm_edge_hulk_0529.py',) 
parse.add_argument('--weight-path', type=str, default='model_10.pth',)  
# parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
cfg_dict = dict(cfg.__dict__)
in_channel = cfg_dict['in_ch']
# define model

if 'net_config' in cfg_dict:
    net = model_factory[cfg.model_type](cfg.n_cats,in_ch=in_channel, aux_mode='eval', net_config=cfg.net_config)
else:
    net = model_factory[cfg.model_type](cfg.n_cats,in_ch=in_channel, aux_mode='eval')
# net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cuda'), strict=False)
# net.eval()
net.cuda()

use_syt_dataset = False
if 'use_syt_dataset' in cfg_dict:
    use_syt_dataset = cfg_dict['use_syt_dataset']

## dataset
if use_syt_dataset:
    # dataset = SytSegmDataset("/home/syt/datasets/ClothSegment/augment/")
    dataset = SytSegmDataset(cfg.train_im_anns, cfg.train_random_crop, cfg_dict['target_size'], cfg_dict['in_ch'])
    # dl = DataLoader(dataset, 16, True, num_workers=8)
    dl = DataLoader(dataset, cfg_dict['ims_per_gpu'], True, num_workers=8, drop_last=True)
    
# traindir = os.path.join(data_path, 'train')
# valdir = os.path.join(data_path, 'val')
# dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)

# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size,
#     sampler=train_sampler, num_workers=4, pin_memory=True)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=batch_size,
#     sampler=test_sampler, num_workers=4, pin_memory=True)

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                    # module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cuda()

# It is a bit slow since we collect histograms on CPU
with torch.no_grad():
    collect_stats(net, dl, num_batches=2)
    compute_amax(net, method="percentile", percentile=99.99)

torch.save(net.state_dict(), 'model_int8.pth')