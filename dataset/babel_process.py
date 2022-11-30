import os
import sys
sys.path.append(os.path.abspath('./'))

import json
from dataset.babel_utils import convert_smplh_path_to_smplx_neutral_path, get_motion_segment
import numpy as np
import pickle
from utils.visualization import render_smplx_body_sequence, frame2video
import smplx
import torch

smplx_folder = '/home/wangzan/Data/SHADE/models_smplx_v1_1/models/'
ROOT = '/home/wangzan/'
l_babel_dense_files = ['train', 'val']
d_folder = ROOT + 'Data/babel/babel_v1.0_release'
amass_smplx_neutral = ROOT + 'Data/amass/smplx_gender_neutral'
motion_seg_folder = ROOT + 'Data/motion/pure_motion/'

DATASETS = ['ACCAD', 'BMLmovi', 'BMLrub', 'CMU', 'EyesJapanDataset', 'MPIHDM05', 'KIT']

def load_segment_by_action(motion_data: dict, action: str):
    p = './dataset/action_segment/{}.json'.format(action)

    if os.path.exists(p):
        print('load previous segment of action {}'.format(action))
        with open(p, 'r') as fp:
            return json.load(fp)
    else:
        segment_data = []
        for key in motion_data.keys():
            segment_data += get_motion_segment(motion_data[key], action)

        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as fp:
            json.dump(segment_data, fp)
        return segment_data

def get_body_vertices_sequence(trans, orient, betas, body_pose, hand_pose):
    seq_len = len(trans)

    body_model = smplx.create(smplx_folder, model_type='smplx',
        gender='neutral', ext='npz',
        num_betas=16,
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=seq_len,
    )

    torch_param = {}
    torch_param['body_pose'] = torch.tensor(body_pose)
    torch_param['betas'] = torch.tensor(betas).reshape(1, -1).repeat(seq_len, 1)
    torch_param['transl'] = torch.tensor(trans)
    torch_param['global_orient'] = torch.tensor(orient)
    torch_param['left_hand_pose'] = torch.tensor(hand_pose[:, 0:45])
    torch_param['right_hand_pose'] = torch.tensor(hand_pose[:, 45:])

    output = body_model(return_verts=True, **torch_param)
    vertices = output.vertices.detach().cpu().numpy()
    joints = output.joints.detach().cpu().numpy()

    return vertices, body_model.faces, joints

def save_segment_annotations(seg: dict, save_folder: str, render: bool=False):
    """ Extract motion segment from original motion annotations

    Args:
        seg: motion segment annotation dict
        save_folder: the save folder of motion segment
        render: if render the motion segment
    """
    seg_id = seg['seg_id']
    babel_sid = seg['babel_sid']
    feat_p = seg['feat_p']
    raw_label =  seg['raw_label']
    start_t = seg['start_t']
    end_t = seg['end_t']
    url = seg['url']
    dur = seg['dur']

    seg_folder = '{:0>6d}'.format(babel_sid) + '_' + seg_id
    save_p = os.path.join(save_folder, seg_folder, 'motion.pkl')
    if os.path.exists(save_p):
        print('exist {}.'.format(save_p))
        return
    else:
        print('generate {}'.format(save_p))

    smplx_neutral_feat_p = convert_smplh_path_to_smplx_neutral_path(feat_p)
    npz_file = os.path.join(amass_smplx_neutral, smplx_neutral_feat_p)
    if not os.path.exists(npz_file):
        print('Without {}'.format(npz_file))
        return
    feat = np.load(npz_file, allow_pickle=True)

    # print(feat.files)
    # print(type(feat))
    # print(feat['gender'])
    # print(feat['surface_model_type'])
    # print(feat['mocap_frame_rate'])
    # print(feat['mocap_time_length'])
    # print(feat['markers_latent'])
    # print(feat['latent_labels'])
    # print(feat['markers_latent_vids'])
    # print(feat['trans'].shape)
    # print(feat['poses'].shape)
    # print(feat['betas'].shape)
    # print(feat['num_betas'])
    # print(feat['root_orient'].shape)
    # print(feat['pose_body'].shape)
    # print(feat['pose_hand'].shape)
    # print(feat['pose_jaw'].shape)
    # print(feat['pose_eye'].shape)

    total_frame = len(feat['trans'])
    ORIGINAL_FRAME_RATE = feat['mocap_frame_rate']
    FRAME_RATIO = int(ORIGINAL_FRAME_RATE / 30)
    start = min(total_frame, max(0, int(ORIGINAL_FRAME_RATE * start_t)))
    end = min(total_frame, max(0, int(ORIGINAL_FRAME_RATE * end_t)))
    assert start < end, 'start frame index is greater than end frame index!'

    indic = list(range(start, end, FRAME_RATIO))
    if indic[-1] >= len(feat['trans']): # some anno have inaccurate time annotation
        raise Exception('Unexcepted start and end indices.')
    
    pkl = (
        feat['gender'], 
        feat['trans'][indic, :].astype(np.float32), 
        feat['root_orient'][indic, :].astype(np.float32), 
        feat['betas'].astype(np.float32), 
        feat['pose_body'][indic, :].astype(np.float32), 
        feat['pose_hand'][indic, :].astype(np.float32), 
        feat['pose_jaw'][indic, :].astype(np.float32), 
        feat['pose_eye'][indic, :].astype(np.float32)
    )

    _, _, joints = get_body_vertices_sequence(pkl[1], pkl[2], pkl[3], pkl[4], pkl[5])
    pkl = (*pkl, joints)
    
    os.makedirs(os.path.dirname(save_p), exist_ok=True)
    with open(save_p, 'wb') as fp:
        pickle.dump(pkl, fp)
    
    if render:
        _, trans, orient, betas, body_pose, hand_pose, _, _, joints = pkl

        render_smplx_body_sequence(
            smplx_folder=smplx_folder, 
            save_folder=os.path.join(save_folder, seg_folder, 'rendering'),
            pkl=(trans, orient, betas, body_pose, hand_pose)
        )
        frame2video(
            path=os.path.join(save_folder, seg_folder, 'rendering/%03d.png'),
            video=os.path.join(save_folder, seg_folder, 'motion.mp4'),
            start=0,
        )
        os.system('rm -rf {}'.format(os.path.join(save_folder, seg_folder, 'rendering/')))

def generate_motion_segment_by_action(action):
    # BABEL Dataset
    babel = {}
    for file in l_babel_dense_files:
        babel.update( json.load(open(os.path.join(d_folder, file+'.json'))) )
    
    ## load satisfying segments
    segment = load_segment_by_action(babel, action)
    print('total segments', len(segment))

    print('** generate {} motion segment **'.format(action))
    for seg in segment:
        set_name = seg['feat_p'].split('/')[0]
        if set_name not in DATASETS: # select motion in valid dataset
            continue
        if seg['end_t'] - seg['start_t'] < 1.0 or seg['end_t'] - seg['start_t'] > 4.0: # select motion range from 1.0s ~ 4.0s
            continue
        
        save_segment_annotations(seg, os.path.join(motion_seg_folder, action), True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str)
    args = parser.parse_args()
    
    generate_motion_segment_by_action(args.action)
