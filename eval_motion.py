import os
import sys
from typing import Any
sys.path.append(os.path.abspath('./'))
import torch
from torch.utils.data import DataLoader
from utils.configuration import parse_args
from utils.utilities import Console, Ploter
from model.solver import MotionSolver
from model.dataloader import MotionDataset, collate_random_motion
import numpy as np

def get_dataloader(args: Any, phase: str):
    shuffle = True
    dataset = MotionDataset(phase, npoints=args.npoints, use_color=args.use_color, use_normal=args.use_normal, max_lang_len=args.max_lang_len, max_motion_len=args.max_motion_len, action=args.action, num_betas=args.num_betas, num_pca_comps=args.num_pca_comps)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_random_motion, num_workers=args.num_workers)
    return dataloader

def test(args):
    args.body_feat_size = args.num_betas
    args.scene_group_size = args.npoints // 256 # need change if scene model changes
    args.input_size = 3 + 6 + 63 # trans + orient + body pose
    test_dataloader = get_dataloader(args, 'val')

    dataloader = {
        'test': test_dataloader
    }

    Console.log('-' * 30)
    Console.log("Test examples: {}".format(len(test_dataloader)))
    
    solver = MotionSolver(args, dataloader)
    torch.manual_seed(0) # fix sample data order

    solver.save_k_sample(k=10)

if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    ## Reproducible
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ## parse input arguments
    args = parse_args()

    ## set logger path
    args.log_dir = os.path.join(args.log_dir, args.stamp)
    if not os.path.isdir(args.log_dir):
        raise Exception('Unexpected log folder')

    ## set cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device("cuda:0" if args.device == 'cuda' else "cpu")
    args.batch_size = 1

    test(args)
    