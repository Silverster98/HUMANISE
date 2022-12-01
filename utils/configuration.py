import argparse
import os
import time

smplx_folder = '/home/wangzan/Data/SHADE/models_smplx_v1_1/models/'
scan2cad_anno = '/home/wangzan/Data/scan2cad/scan2cad_download_link/full_annotations.json'
scannet_folder = '/home/wangzan/Data/scannet/scans/'
referit3d_sr3d = '/home/wangzan/Data/referit3d/sr3d.csv'

pure_motion_folder = '/home/wangzan/Data/motion/pure_motion/'
align_data_folder = '/home/wangzan/Data/motion/align_data_release/'
preprocess_scene_folder = '/home/wangzan/Projects/Pointnet2.ScanNet/preprocessing/scannet_scenes'


## train & test
batch_size = 16
learning_rate = 1e-4
num_epoch = 500
device = 'cuda'
resume_model = ''
num_workers = 0
weight_loss_rec = 1.0
weight_loss_rec_pose = 1.0
weight_loss_rec_vertex = 1.0
weight_loss_kl = 0.1
weight_loss_vposer = 1e-3
weight_loss_ground = 1.0
action = 'sit'

## model setting
pretrained_scene_model = ''
lang_feat_size = 768
scene_feat_size = 512
scene_group_size = 16 # pointnet++ final output size
max_lang_len = 32
max_motion_len = 120
model_hidden_size = 512
model_condition_size = 512
model_z_latent_size = 32
npoints = 8192

## smplx
num_pca_comps = 12
num_betas = 10
gender = 'neutral'


def parse_args():
    parser = argparse.ArgumentParser()

    ## path setting
    parser.add_argument('--log_dir', 
                        type=str, 
                        default=os.path.join(os.getcwd(), 'logs/'),
                        help='dir for saving checkpoints and logs')
    parser.add_argument('--stamp', 
                        type=str, 
                        default=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                        help='timestamp')
    ## train & test setting
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    parser.add_argument('--device', 
                        type=str, 
                        default=device,
                        help='set device for training')
    parser.add_argument('--resume_model',
                        type=str,
                        default=resume_model,
                        help='resume model path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=num_workers,
                        help='number of dataloader worker processer')
    parser.add_argument('--weight_loss_rec',
                        type=float,
                        default=weight_loss_rec,
                        help='loss weight of rec loss')
    parser.add_argument('--weight_loss_rec_pose',
                        type=float,
                        default=weight_loss_rec_pose,
                        help='loss weight of rec pose loss')
    parser.add_argument('--weight_loss_rec_vertex',
                        type=float,
                        default=weight_loss_rec_vertex,
                        help='loss weight of rec vertex loss')
    parser.add_argument('--weight_loss_kl',
                        type=float,
                        default=weight_loss_kl,
                        help='loss weight of kl loss')
    parser.add_argument('--weight_loss_vposer',
                        type=float,
                        default=weight_loss_vposer,
                        help='loss weight of vposer loss')
    parser.add_argument('--weight_loss_ground',
                        type=float,
                        default=weight_loss_ground,
                        help='loss weight of ground loss')
    parser.add_argument('--all_body_vertices',
                        action="store_true",
                        help='use all body vertices to regress')
    parser.add_argument('--action',
                        type=str,
                        default=action,
                        help='action type')
    ## model setting
    parser.add_argument('--pretrained_scene_model',
                        type=str,
                        default=pretrained_scene_model,
                        help='pre-trained scene model')
    parser.add_argument('--lang_feat_size',
                        type=int,
                        default=lang_feat_size,
                        help='language feature size')
    parser.add_argument('--scene_feat_size',
                        type=int,
                        default=scene_feat_size,
                        help='scene feature size')
    parser.add_argument('--scene_group_size',
                        type=int,
                        default=scene_group_size,
                        help='scene group size')
    parser.add_argument('--use_color',
                        action="store_true",
                        help='use point rgb color')
    parser.add_argument('--use_normal',
                        action="store_true",
                        help='use point normal')
    parser.add_argument('--max_lang_len',
                        type=int,
                        default=max_lang_len,
                        help='max length of language description')
    parser.add_argument('--max_motion_len',
                        type=int,
                        default=max_motion_len,
                        help='max length of motion sequence')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=model_hidden_size,
                        help='the size the hidden state in CVAE model')
    parser.add_argument('--condition_latent_size',
                        type=int,
                        default=model_condition_size,
                        help='the size the condition latent')
    parser.add_argument('--z_size',
                        type=int,
                        default=model_z_latent_size,
                        help='the size the z latent')
    parser.add_argument('--npoints',
                        type=int,
                        default=npoints,
                        help='sample points number of pointcloud')
    ## smplx setting
    parser.add_argument('--num_pca_comps', 
                        type=int, 
                        default=num_pca_comps,
                        help='number of pca component of hand pose')
    parser.add_argument('--num_betas', 
                        type=int, 
                        default=num_betas,
                        help='number of pca component of body shape beta')
    
    args = parser.parse_args()

    return args
