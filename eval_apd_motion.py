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
import sklearn
from utils.smplx_util import SMPLX_Util, marker_indic
import utils.configuration as config

def get_dataloader(args: Any, phase: str):
    shuffle = True
    dataset = MotionDataset(phase, npoints=args.npoints, use_color=args.use_color, use_normal=args.use_normal, max_lang_len=args.max_lang_len, max_motion_len=args.max_motion_len, action=args.action, num_betas=args.num_betas, num_pca_comps=args.num_pca_comps)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_random_motion, num_workers=args.num_workers)
    return dataloader

def test(args, dataset='val'):
    args.body_feat_size = args.num_betas
    args.scene_group_size = args.npoints // 256 # need change if scene model changes
    args.input_size = 3 + 6 + 63 # trans + orient + body pose
    test_dataloader = get_dataloader(args, dataset)

    dataloader = {
        'test': test_dataloader
    }

    Console.log('-' * 30)
    Console.log("Test examples: {}".format(len(test_dataloader)))
    
    solver = MotionSolver(args, dataloader)
    torch.manual_seed(0) # fix sample data order


    ## pairwise distance
    k=20
    write_file=os.path.join(args.log_dir, 'metric/pairwise_dist.json')

    solver._set_phase('val')
    solver.config.resume_model = os.path.join(solver.config.log_dir, 'model_best.pth')
    solver._load_state_dict()

    with torch.no_grad():
        pdist = []
        for i, data in enumerate(solver.dataloader['test']):
            [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
            trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

            nframe = torch.sum(~motion_mask)
            [pred_target_object, action_logits, rec_trans, rec_orient, rec_pose_body, rec_mask, atten_score, scene_xyz] = solver._sample(
                point_coords, point_feats, offset, lang_desc, lang_mask, betas, k, 'fixed', nframe, trans, orient, pose_body
            )

            mask = ~rec_mask[0][0]
            n = mask.sum()
            trans_sample = rec_trans[0, :, mask, :].reshape(k * n, -1)
            orient_sample = rec_orient[0, :, mask, :].reshape(k * n, -1)
            pose_body_sample = rec_pose_body[0, :, mask, :].reshape(k * n, -1)
            pose_hand = torch.zeros(k, n, pose_hand.shape[-1]).reshape(k * n, -1)

            pkl = solver._convert_compute_smplx_to_render_smplx((
                trans_sample, orient_sample, betas, pose_body_sample, pose_hand))
            
            body_vertices, body_faces, body_joints = SMPLX_Util.get_body_vertices_sequence(
                config.smplx_folder, pkl, num_betas=solver.config.num_betas)
            _, V, D = body_vertices.shape
            body_vertices = body_vertices.reshape(k, n, V, D)[:, :, marker_indic, :]
            body_feature = body_vertices.reshape(k, n, -1)
            dist = 0
            for j in range(n):
                F = body_feature[:, j, :]
                dist += (sklearn.metrics.pairwise_distances(F, F, metric='l2').sum() / (k * (k - 1)))
            
            pdist.append(dist / n.item())
            
        results = {}
        results['average_pairwise_dist'] = sum(pdist) / len(pdist)
        results['pairwise_dist'] = pdist

        os.makedirs(os.path.dirname(write_file), exist_ok=True)
        import json
        with open(write_file, 'w') as fp:
            json.dump(results, fp)
    
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

    ## training
    test(args, dataset='val')
    