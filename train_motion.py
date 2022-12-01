import os
import sys
from typing import Any
sys.path.append(os.path.abspath('./'))
from loguru import logger
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.configuration import parse_args
from utils.utilities import Console, Ploter
from model.solver import MotionSolver
from model.dataloader import MotionDataset, collate_random_motion

def get_dataloader(args: Any, phase: str):
    shuffle = True if phase == 'train' else False
    dataset = MotionDataset(phase, npoints=args.npoints, use_color=args.use_color, use_normal=args.use_normal, max_lang_len=args.max_lang_len, max_motion_len=args.max_motion_len, action=args.action, num_betas=args.num_betas, num_pca_comps=args.num_pca_comps)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_random_motion, num_workers=args.num_workers, drop_last=True)
    return dataloader

def train(args):
    """ training portal

    Args:
        args: config arguments
    """
    args.body_feat_size = args.num_betas
    args.scene_group_size = args.npoints // 256 # need change if scene model changes
    args.input_size = 3 + 6 + 63 # trans + orient + body pose

    train_dataloader= get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, 'val')

    dataloader = {
        'train': train_dataloader,
        'val': val_dataloader,
    }

    Console.log('-' * 30)
    Console.log('\n[Info]')
    Console.log("Train examples: {}".format(len(train_dataloader)))
    Console.log("Eval examples: {}".format(len(val_dataloader)))

    solver = MotionSolver(args, dataloader)
    # solver.check_data()
    Console.log('Start training...\n')
    solver()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
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
    os.makedirs(args.log_dir, exist_ok=True)
    ## set tensorboard and text logger
    logger.add(os.path.join(args.log_dir, 'runtime.log'), format='{time:YYYY-MM-DD HH:mm:ss.SSS} {level} {message}')
    writer = SummaryWriter(log_dir=args.log_dir)

    Console.setPrinter(printer=logger.info, debug=False)
    Ploter.setWriter(writer=writer)
    
    Console.log('Init logger...')
    Console.log('[************ Global Configuration ************] \n' + str(args) + '\n')

    ## set cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.device = torch.device("cuda:0" if args.device == 'cuda' else "cpu")

    train(args)
