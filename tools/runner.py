import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156", #plane
        # "04379243",  #table
        # "03790512", #motorbike
        # "03948459", #pistol
        # "03642806", #laptop
        # "03467517",     #guitar
        # "03261776", #earphone
        # "03001627", #chair
        # "02958343", #car
        # "04090263", #rifle
        # "03759954", # microphone
    ]
    with torch.no_grad():
        loss = []
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            # if  taxonomy_ids[0] not in useful_cate:
            #     continue
            # if taxonomy_ids[0] == "02691156":
            #     a, b= 90, 135
            # elif taxonomy_ids[0] == "04379243":
            #     a, b = 30, 30
            # elif taxonomy_ids[0] == "03642806":
            #     a, b = 30, -45
            # elif taxonomy_ids[0] == "03467517":
            #     a, b = 0, 90
            # elif taxonomy_ids[0] == "03261776":
            #     a, b = 0, 75
            # elif taxonomy_ids[0] == "03001627":
            #     a, b = 30, -45
            # else:
            #     a, b = 0, 0


            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            loss1 = base_model(points, vis=True)
            loss.append(loss1)
  
            # data_path = f'./vis_no_rotate_center/{taxonomy_ids[0]}/{taxonomy_ids[0]}_{idx}'
            # if not os.path.exists(data_path):
            #     os.makedirs(data_path)

            # vis_points = points.reshape(-1,3).detach().cpu().numpy() 
            # print(vis_points.shape, "vis shape")
            # vis_points = misc.get_ptcloud_img(vis_points,a,b)
            # final_image = vis_points
            # img_path = os.path.join(data_path, f'input.jpg')
            # cv2.imwrite(img_path, final_image)
            
           

            # if idx > 100:   # Sepcify the number of images to iterate through
            #     break
        print(len(loss))
        print(torch.vstack(loss).mean(dim=0))

        return
