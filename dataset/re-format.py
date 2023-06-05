import os, sys
import argparse
import csv
import glob
import pickle
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion as Q
from trimesh import transform_points
from natsort import natsorted


class HUMANISE():
    def __init__(self, data_dir: str, save_dir: str) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.save_motion_dir = os.path.join(self.save_dir, 'motions')
        self.num_betas = 10
        
        self.annotation_fields = ['motion_id', 'scene_id', 'scene_trans_x', 'scene_trans_y', 'scene_trans_z', 'object_id', 'object_label', 'object_semantic_label', 'action', 'text']
        self.annotation_data = []
    
    def process(self) -> None:
        print(f'Processing HUMANISE dataset in {self.data_dir} ...')

        os.makedirs(self.save_motion_dir, exist_ok=True)
        aligns = natsorted(glob.glob(os.path.join(self.data_dir, 'align_data_release', '*', '*', 'anno.pkl')))
        
        anno_list = []
        motion_data = {}
        for align in aligns:
            with open(align, 'rb') as f:
                data = pickle.load(f)
            
            for i, p in enumerate(data):
                anno_list.append(p)

                action = p['action']
                motion_id = p['motion']
                if motion_id not in motion_data:
                    with open(os.path.join(self.data_dir, 'pure_motion', action, motion_id, 'motion.pkl'), 'rb') as fp:
                        mdata = pickle.load(fp)
                    motion_data[motion_id] = mdata
        
        for anno_index in tqdm(range(len(anno_list))):
            anno_data = anno_list[anno_index]
            
            motion_id = anno_data['motion']
            motion_trans = anno_data['translation']
            motion_rotat = anno_data['rotation']

            anchor_frame_index = {'sit': -1, 'stand up': 0, 'walk': -1, 'lie': -1}[anno_data['action']]
            gender, origin_trans, origin_orient, betas, pose_body, pose_hand, \
                pose_jaw, pose_eye, joints_traj = motion_data[motion_id]
            
            cur_trans, cur_orient, cur_pelvis = _transform_smplx_from_origin_to_sampled_position(
                motion_trans, motion_rotat, origin_trans, origin_orient, joints_traj[:, 0, :], anchor_frame_index)
            betas = betas[:self.num_betas]

            param_seq = np.concatenate([cur_trans, cur_orient, pose_body, pose_hand], axis=-1)
            # print(param_seq.shape, betas.shape)
            # print(param_seq.dtype, betas.dtype)
            with open(os.path.join(self.save_motion_dir, f'{anno_index:06d}.pkl'), 'wb') as fp:
                pickle.dump((param_seq, betas), fp)

            self.annotation_data.append([
                f"{anno_index:06d}",
                f"{anno_data['scene']}",
                f"{anno_data['scene_translation'][0]:.8f}",
                f"{anno_data['scene_translation'][1]:.8f}",
                f"{anno_data['scene_translation'][2]:.8f}",
                f"{anno_data['object_id']}",
                f"{anno_data['object_label']}",
                f"{anno_data['object_semantic_label']}",
                f"{anno_data['action']}",
                f"{anno_data['utterance']}",
            ])
        
        with open(os.path.join(self.save_dir, 'annotation.csv'), 'w') as fp:
            csvwriter = csv.writer(fp)
            csvwriter.writerow(self.annotation_fields)
            csvwriter.writerows(self.annotation_data)

def _convert_smplx_verts_transfomation_matrix_to_body(T, trans, orient, pelvis):
    """ Convert transformation to smplx trans and orient

    Args:
        T: target transformation matrix
        trans: origin trans of smplx parameters (a sequence)
        orient: origin orient of smplx parameters (a sequence)
        pelvis: origin pelvis sequence
    
    Return:
        Transformed trans and orient smplx parameters
    """
    R = T[0:3, 0:3]
    t = T[0:3, -1]

    pelvis = pelvis - trans
    trans = np.matmul(R, trans + pelvis) - pelvis
    orient = np.matmul(R, Q(axis=orient/np.linalg.norm(orient), angle=np.linalg.norm(orient)).rotation_matrix)
    orient = Q(matrix=orient)
    orient = orient.axis * orient.angle
    return trans + t, orient.astype(np.float32)

def _transform_smplx_from_origin_to_sampled_position(
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
            t_, o_ = _convert_smplx_verts_transfomation_matrix_to_body(T, origin_trans[i], origin_orient[i], origin_pelvis[i])
            trans_t.append(t_)
            orient_t.append(o_)
        
        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = transform_points(origin_pelvis, T)
        return trans_t, orient_t, pelvis_t


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./data')
args = parser.parse_args()

HUMANISE(args.data_dir, args.save_dir).process()
