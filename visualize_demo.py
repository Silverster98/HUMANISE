import os, sys
import argparse
import random
import glob
import smplx
import torch
import pickle
import trimesh
import pandas as pd
import numpy as np
from natsort import natsorted
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--scene_dir', type=str, default='./data')
parser.add_argument('--smplx_dir', type=str, default='./data')
args = parser.parse_args()

body_model_neutral = smplx.create(
        args.smplx_dir, model_type='smplx',
        gender='neutral', ext='npz',
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
        batch_size=1,
).to(device='cpu')

@torch.no_grad()
def smplx_params_to_meshes(transl, orient, betas, body_pose, hand_pose):
    verts = []
    for i in range(len(transl)):
        output = body_model_neutral(
            return_verts=True,
            transl=transl[i:i+1],
            global_orient=orient[i:i+1],
            betas=betas,
            body_pose=body_pose[i:i+1],
            left_hand_pose=hand_pose[i:i+1, :45],
            right_hand_pose=hand_pose[i:i+1, 45:]
        )
        vertices = output.vertices.detach().cpu().numpy()
        verts.append(vertices)

    verts = np.concatenate(verts, axis=0)
    faces = body_model_neutral.faces

    return verts, faces

def load_humanise_motions(motion_path, annotations, scene_dir):
    index = int(motion_path.split('/')[-1].split('.')[0])
    poses, betas = pickle.load(open(motion_path, 'rb'))

    trans = torch.from_numpy(poses[:, :3]).float()
    root_orient = torch.from_numpy(poses[:, 3:6]).float()
    pose_body = torch.from_numpy(poses[:, 6:69]).float()
    pose_hand = torch.from_numpy(poses[:, 69:]).float()
    betas = torch.from_numpy(betas).float().unsqueeze(0)

    verts, faces = smplx_params_to_meshes(trans, root_orient, betas, pose_body, pose_hand)

    texts = [annotations.loc[index]['text']]

    scene_id = annotations.loc[index]['scene_id']
    scene_trans = np.array([
        float(annotations.loc[index]['scene_trans_x']),
        float(annotations.loc[index]['scene_trans_y']),
        float(annotations.loc[index]['scene_trans_z']),
    ], dtype=np.float32)
    scene = trimesh.load(os.path.join(scene_dir, f'{scene_id}/{scene_id}_vh_clean_2.ply'), process=False)
    ## if you have preporcessed scene point cloud, you can use also it to visualize the scene
    # scene_pcd = np.load(os.path.join(scene_dir, '{scene_id}.npy')) # <x,y,z,r,g,b,nx,ny,nz,inst,sem>
    # scene_pos = scene_pcd[:, 0:3]
    # scene_col = scene_pcd[:, 3:6].astype(np.uint8)
    # scene_col[scene_pcd[:, -2] == int(annotations.loc[index]['object_id']), :] = np.array([255, 0, 0], dtype=np.uint8)
    # scene = trimesh.PointCloud(vertices=scene_pos, colors=scene_col)
    scene.apply_translation(scene_trans)

    return verts, faces, texts, scene


## Visualize
motion_paths = natsorted(glob.glob(os.path.join(args.data_dir, 'motions/*.pkl')))
anno_file = pd.read_csv(os.path.join(args.data_dir, 'annotation.csv'))
total_amount = anno_file.shape[0]
assert total_amount == len(motion_paths)
random.shuffle(motion_paths)

for motion_path in motion_paths:    
    verts, faces, texts, scene = load_humanise_motions(motion_path, anno_file, args.scene_dir)

    S = trimesh.Scene()
    S.add_geometry(scene)
    S.add_geometry(trimesh.creation.axis())
    for i in range(0, verts.shape[0], 10): # load body mesh every 10 frames
        S.add_geometry(trimesh.Trimesh(vertices=verts[i], faces=faces)) 
    print(texts)
    S.show()
