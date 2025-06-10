import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import CLIP
import copy
import torchvision.transforms as transforms
from Dual_prompt.no_blip2.modal import DoubleEncoder
from Dual_prompt.no_blip2.config_no_blip2 import ex_test


@ex_test.add_config({
    'use_prompt':True,

    'train_type':'VTPT3',

    #Prompt
    'embed_dim':768,
    "prompt_pool": True,
    "pool_size": 20,
    "prompt_length": 1,
    "top_k": 3,
    'prompt_init': 'uniform',
    'prompt_key': True,
    'prompt_key_init': 'uniform',
    'batchwise_prompt': False,
    'embedding_key': 'cls',
})
def get_transform():
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

@ex_test.automain
def main(_config):
    print(_config)
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"], workers=True)
    model = DoubleEncoder(_config)
    model.cuda()
    ckpt = torch.load(_config['checkpoint'])
    model.load_state_dict(ckpt['state_dict'])
    devices=model.devices
    transform=get_transform()
    model.visual.transformer