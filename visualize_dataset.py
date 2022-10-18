import numpy as np
import argparse
import pickle
import trimesh
import os
import glob
from  natsort import natsorted
from trimesh import transform_points
from pyquaternion import Quaternion as Q

import utils.configuration as config
from utils.smplx_util import SMPLX_Util
from utils.visualization import frame2video, render_motion_in_scene

def transform_smplx_from_origin_to_sampled_position(
        sampled_trans: np.ndarray,
        sampled_rotat: np.ndarray,
        origin_trans: np.ndarray,
        origin_orient: np.ndarray,
        origin_pelvis: np.ndarray,
        anchor_frame: int=0,
    ):
        """ Convert original smplx parameters to transformed smplx parameters

        Args:
            sampled_trans: sampled valid position
            sampled_rotat: sampled valid rotation
            origin_trans: original trans param array
            origin_orient: original orient param array
            origin_pelvis: original pelvis trajectory
            anchor_frame: the anchor frame index for transform motion, this value is very important!!!
        
        Return:
            Transformed trans, Transformed orient, Transformed pelvis
        """
        position = sampled_trans
        rotat = sampled_rotat

        T1 = np.eye(4, dtype=np.float32)
        T1[0:2, -1] = -origin_pelvis[anchor_frame, 0:2]
        T2 = Q(axis=[0, 0, 1], angle=rotat).transformation_matrix.astype(np.float32)
        T3 = np.eye(4, dtype=np.float32)
        T3[0:3, -1] = position
        T = T3 @ T2 @ T1

        trans_t = []
        orient_t = []
        for i in range(len(origin_trans)):
            t_, o_ = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(T, origin_trans[i], origin_orient[i], origin_pelvis[i])
            trans_t.append(t_)
            orient_t.append(o_)
        
        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = transform_points(origin_pelvis, T)
        return trans_t, orient_t, pelvis_t

def get_anchor_frame_index(action: str):
    action_anchor = {
        'sit': -1,
        'stand up': 0,
        'walk': -1,
        'lie': -1,
    }
    return action_anchor[action]

def visualize_result(path, index=0, vis=False, save_folder=None, del_imgs=True):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        print('there are {} cases in this anno pkl file.'.format(len(data)))
    p = data[index]
    action = p['action']
    scene_id = p['scene']
    motion_id = p['motion']
    scene_T = p['scene_translation']
    motion_trans = p['translation']
    motion_rotat = p['rotation']
    utterance = p['utterance']
    anchor_frame = get_anchor_frame_index(action)

    scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))
    static_scene = trimesh.load(scene_path, process=False)
    static_scene.apply_translation(scene_T)

    motion_path = os.path.join(config.pure_motion_folder, action, motion_id, 'motion.pkl')
    with open(motion_path, 'rb') as fp:
        motion_data = pickle.load(fp)
    
    ## transform motion
    _, trans, orient, betas, pose_body, pose_hand, _, _, joints = motion_data
    trans, orient, pelvis = transform_smplx_from_origin_to_sampled_position(motion_trans, motion_rotat, trans, orient, joints[:, 0, :], anchor_frame)
    betas = betas[:10]

    body_vertices, body_faces, _ = SMPLX_Util.get_body_vertices_sequence(
        config.smplx_folder, 
        (trans, orient, betas, pose_body, pose_hand),
        num_betas=10
    )

    ## print utterance
    print('visualized case index is {}, and its description is {}'.format(index, utterance))
    print('rendering...')
    
    # visualize with trimesh
    if vis:
        S =trimesh.Scene()
        for i in range(0, len(body_vertices), 5):
            S.add_geometry(
                trimesh.Trimesh(vertices=body_vertices[i], faces=body_faces)
            )
        S.add_geometry(static_scene)
        S.show()

    ## rendering mp4
    save_folder = os.path.dirname('./tmp/') if save_folder is None else save_folder
    render_motion_in_scene(
        smplx_folder=config.smplx_folder,
        save_folder=os.path.join(save_folder, 'imgs/'),
        pkl=(trans, orient, betas, pose_body, pose_hand),
        scene_mesh=static_scene,
        auto_camera=False,
        num_betas=10
    )
    frame2video(
        path=os.path.join(save_folder, 'imgs/%03d.png'),
        video=os.path.join(save_folder, '{:0>3d}.mp4'.format(index)),
        start=0,
    )
    if del_imgs:
        os.system("rm -rf {}".format(os.path.join(save_folder, 'imgs')))
    
    print('done.')

def visualize_and_save_all(action, save_folder):
    pkls = natsorted(glob.glob(os.path.join(config.align_data_folder, action, '*/anno.pkl')))
    save_folder = os.path.join(save_folder, action)

    for pkl in pkls:
        with open(pkl, 'rb') as fp:
            data = pickle.load(fp)
            ncase = len(data)
        
        uid = pkl.split('/')[-2]
        sdir = os.path.join(save_folder, uid)

        if int(uid[5:9]) > 20:
            break
        
        os.makedirs(sdir, exist_ok=True)
        print(f'rendering video in {sdir}')
        for i in range(ncase):
            visualize_result(pkl, index=i, save_folder=sdir)

def render_one_case():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pkl', type=str)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--vis', action="store_true")
    args = parser.parse_args()

    visualize_result(args.pkl, args.index, args.vis)

def render_all_cases():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str)
    parser.add_argument('--saved', type=str)
    args = parser.parse_args()

    visualize_and_save_all(args.action, args.saved)

if __name__ == '__main__':
    # render_all_cases()
    render_one_case()
