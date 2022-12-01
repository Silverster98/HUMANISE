from typing import Tuple
import numpy as np
import smplx
import torch
import os
import trimesh
import pyrender
from PIL import Image
from pyquaternion import Quaternion
from natsort import natsorted
import glob
from utils.smplx_util import SMPLX_Util
from sklearn.neighbors import KDTree
import cv2 as cv

def render_smplx_body_sequence(
    smplx_folder: str,
    save_folder: str,
    pkl: tuple=None, 
    trans_: np.ndarray=None, 
    orient_: np.ndarray=None, 
    betas_: np.ndarray=None, 
    body_pose_: np.ndarray=None, 
    hand_pose_: np.ndarray=None,
    H: int=512, 
    W: int=512,
):
    """ Render smplx body motion
    """
    if pkl is not None:
        trans, orient, betas, body_pose, hand_pose = pkl
    else:
        trans, orient, betas, body_pose, hand_pose = trans_, orient_, betas_, body_pose_, hand_pose_
    
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
    
    for i in range(len(vertices)):
        body = trimesh.Trimesh(vertices[i], body_model.faces, process=False)
            
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        body_mesh = pyrender.Mesh.from_trimesh(
            body, material=material)

        scene = pyrender.Scene()
        camera = pyrender.camera.IntrinsicsCamera(
            fx=550, fy=550,
            cx=340, cy=256)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        camera_pose = Quaternion(axis=[0, 0, 1], angle=np.pi / 2).transformation_matrix @ Quaternion(axis=[1, 0, 0], angle=np.pi / 2).transformation_matrix @ Quaternion(axis=[1, 0, 0], angle=-np.pi/12).transformation_matrix

        x_body, y_body, z_body = trans[i]
        camera_pose[0:3, -1] += np.array([x_body+3, y_body, 1.5])
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh, 'mesh')

        axis_mesh = trimesh.creation.axis(origin_size=0.02)
        ground_mesh = trimesh.creation.cylinder(radius=10, height=0.2, transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.1], [0, 0, 0, 1]], dtype=np.float32))
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
        scene.add(pyrender.Mesh.from_trimesh(ground_mesh, smooth=False), 'mesh_ground')

        r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        os.makedirs(save_folder, exist_ok = True)
        img.save(os.path.join(save_folder, '{:0>3d}.png'.format(i)))
        r.delete()

def render_motion_in_scene(
    smplx_folder: str,
    save_folder: str,
    scene_mesh: trimesh.Trimesh=None,
    pkl: tuple=None, 
    trans_: np.ndarray=None, 
    orient_: np.ndarray=None, 
    betas_: np.ndarray=None, 
    body_pose_: np.ndarray=None, 
    hand_pose_: np.ndarray=None,
    auto_camera: bool=False,
    cam_pose: np.ndarray=None,
    H: int=1080, 
    W: int=1920,
    num_betas: int=16,
):
    """ Render smplx body motion
    """
    if pkl is not None:
        trans, orient, betas, body_pose, hand_pose = pkl
    else:
        trans, orient, betas, body_pose, hand_pose = trans_, orient_, betas_, body_pose_, hand_pose_
    
    seq_len = len(trans)
    body_model = smplx.create(smplx_folder, model_type='smplx',
        gender='neutral', ext='npz',
        num_betas=num_betas,
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
    pelvis_xy = output.vertices.detach().cpu().numpy()[:, 0, 0:2]

    if auto_camera:
        ## compute a proper camera pose
        traj_center = (pelvis_xy[0] + pelvis_xy[-1]) * 0.5
        traj_vec = (pelvis_xy[-1] - pelvis_xy[0])
        cam_vec = Quaternion(axis=[0, 0, 1], angle=np.pi / 2).rotation_matrix[0:2, 0:2] @ traj_vec
        cam_vec = np.array([*cam_vec, 0])
        cam_vec_init = np.array([0, 1., 0])
        a = np.arccos(np.dot(cam_vec, cam_vec_init) / np.linalg.norm(cam_vec))
        if np.cross(cam_vec, cam_vec_init)[-1] < 0:
            a = np.pi * 2 - a
        camera_pose = Quaternion(axis=[0, 0, 1], angle=-a).transformation_matrix @ Quaternion(axis=[1, 0, 0], angle=np.pi * 5 / 12).transformation_matrix
        cam_xy = np.array([*traj_center, 0]) - cam_vec / np.linalg.norm(cam_vec) * 2
        camera_pose[0:3, -1] = cam_xy + np.array([0, 0, 1.5])
    else:
        ## use fixed top-down view
        scene_center = scene_mesh.vertices.mean(axis=0)
        camera_pose = np.eye(4)
        camera_pose[0:3, -1] = np.array([*scene_center[0:2], 7])
    
    if cam_pose is not None:
        camera_pose = cam_pose

    ## rendering
    for i in range(len(vertices)):
        body = trimesh.Trimesh(vertices[i], body_model.faces, process=False)
            
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        body_mesh = pyrender.Mesh.from_trimesh(
            body, material=material)

        scene = pyrender.Scene()
        camera = pyrender.camera.IntrinsicsCamera(
            fx=1060, fy=1060,
            cx=951.30, cy=536.77)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        scene.add(body_mesh, 'mesh')

        axis_mesh = trimesh.creation.axis(origin_size=0.02)
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh, smooth=False), 'mesh_scene')

        r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        os.makedirs(save_folder, exist_ok = True)
        img.save(os.path.join(save_folder, '{:0>3d}.png'.format(i)))
        r.delete()

def render_reconstructed_motion_in_scene(
    smplx_folder: str,
    save_folder: str,
    pkl_rec: Tuple,
    scene_mesh: trimesh.Trimesh,
    pkl_gt: Tuple=None,
    cam_pose: np.ndarray=None,
    H: int=1080, 
    W: int=1920,
    num_betas: int=16,
):
    """ Render smplx body motion
    """
    
    body_vertices_rec, body_faces, _ = SMPLX_Util.get_body_vertices_sequence(smplx_folder, pkl_rec, num_betas=num_betas)
    if pkl_gt is not None:
        body_vertices_gt, _, _ = SMPLX_Util.get_body_vertices_sequence(smplx_folder, pkl_gt, num_betas=num_betas)
    else:
        body_vertices_gt = None

    ## rendering
    material_green = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 1.0, 0.0, 1.0))
    material_blue = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 0.0, 1.0, 1.0))
    
    scene = pyrender.Scene()

    if cam_pose is not None:
        camera_pose = cam_pose
    else:
        ## use fixed top-down view
        scene_center = scene_mesh.vertices.mean(axis=0)
        camera_pose = np.eye(4)
        camera_pose[0:3, -1] = np.array([*scene_center[0:2], 7])


    ## rendering
    for i in range(len(body_vertices_rec)):
        body_rec = trimesh.Trimesh(body_vertices_rec[i], body_faces, process=False)
        
        if body_vertices_gt is not None and i < len(body_vertices_gt):
            body_gt = trimesh.Trimesh(body_vertices_gt[i], body_faces, process=False)
        else:
            body_gt = None
        
        scene = pyrender.Scene()
        body_mesh_rec = pyrender.Mesh.from_trimesh(body_rec, material=material_blue)
        scene.add(body_mesh_rec)
        if body_gt is not None:
            body_mesh_gt = pyrender.Mesh.from_trimesh(body_gt, material=material_green)
            scene.add(body_mesh_gt)

        camera = pyrender.camera.IntrinsicsCamera(
            fx=1060, fy=1060,
            cx=951.30, cy=536.77)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        axis_mesh = trimesh.creation.axis(origin_size=0.02)
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh, smooth=False), 'mesh_scene')

        r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        os.makedirs(save_folder, exist_ok = True)
        img.save(os.path.join(save_folder, '{:0>3d}.png'.format(i)))
        r.delete()

def render_sample_k_motion_in_scene(
    smplx_folder: str,
    save_folder: str,
    pkl_rec: Tuple,
    scene_mesh: trimesh.Trimesh,
    pkl_gt: Tuple=None,
    cam_pose: np.ndarray=None,
    H: int=1080, 
    W: int=1920,
    num_betas: int=16,
):
    """ Render smplx body motion
    """
    trans, orient, betas, pose_body, pose_hand = pkl_rec
    K, S, _ = trans.shape
    body_vertices_rec, body_faces, _ = SMPLX_Util.get_body_vertices_sequence(
        smplx_folder, 
        (
            trans.reshape(K * S, -1),
            orient.reshape(K * S, -1),
            betas,
            pose_body.reshape(K * S, -1),
            pose_hand.reshape(K * S, -1),
        ),
        num_betas=num_betas
    )
    body_vertices_rec = body_vertices_rec.reshape(K, S, -1, 3)

    if pkl_gt is not None:
        body_vertices_gt, _, _ = SMPLX_Util.get_body_vertices_sequence(smplx_folder, pkl_gt, num_betas=num_betas)
    else:
        body_vertices_gt = None

    ## rendering
    material_green = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 1.0, 0.0, 1.0))
    material_blue = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.0, 0.0, 1.0, 1.0))
    
    scene = pyrender.Scene()

    if cam_pose is not None:
        camera_pose = cam_pose
    else:
        ## use fixed top-down view
        scene_center = scene_mesh.vertices.mean(axis=0)
        camera_pose = np.eye(4)
        camera_pose[0:3, -1] = np.array([*scene_center[0:2], 7])

    ## rendering
    for i in range(S):
        scene = pyrender.Scene()

        for k in range(K):
            body_rec = trimesh.Trimesh(body_vertices_rec[k][i], body_faces, process=False)
            body_mesh_rec = pyrender.Mesh.from_trimesh(body_rec, material=material_blue)
            scene.add(body_mesh_rec)
        
        if body_vertices_gt is not None and i < len(body_vertices_gt):
            body_gt = trimesh.Trimesh(body_vertices_gt[i], body_faces, process=False)
            body_mesh_gt = pyrender.Mesh.from_trimesh(body_gt, material=material_green)
            scene.add(body_mesh_gt)

        camera = pyrender.camera.IntrinsicsCamera(
            fx=1060, fy=1060,
            cx=951.30, cy=536.77)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        axis_mesh = trimesh.creation.axis(origin_size=0.02)
        scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
        scene.add(pyrender.Mesh.from_trimesh(scene_mesh, smooth=False), 'mesh_scene')

        r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
        color, _ = r.render(scene)
        color = color.astype(np.float32) / 255.0
        img = Image.fromarray((color * 255).astype(np.uint8))
        os.makedirs(save_folder, exist_ok = True)
        img.save(os.path.join(save_folder, '{:0>3d}.png'.format(i)))
        r.delete()

def get_multi_colors_by_hsl(begin_color, end_color, coe):
    begin_color = begin_color.reshape(1,1,3).repeat(len(coe), axis=1)
    begin_rgb = begin_color / 255
    begin_hls = cv.cvtColor(np.array(begin_rgb, dtype=np.float32), cv.COLOR_RGB2HLS)
    end_color = end_color.reshape(1,1,3).repeat(len(coe), axis=1)
    end_rgb = end_color / 255
    end_hls = cv.cvtColor(np.array(end_rgb, dtype=np.float32), cv.COLOR_RGB2HLS)

    hls = ((end_hls - begin_hls) * coe.reshape(-1, 1).repeat(3, axis=1) + begin_hls)
    rgb = cv.cvtColor(np.array(hls, dtype=np.float32), cv.COLOR_HLS2RGB)
    return (rgb*255).astype(np.uint8).reshape(-1, 3)

def render_attention(
    save_folder: str,
    scene_mesh: trimesh.Trimesh,
    atten_score: np.ndarray,
    atten_pos: np.ndarray,
    pred_target_object: np.ndarray=None,
    cam_pose: np.ndarray=None,
    H: int=1080, 
    W: int=1920,
):
    """ Render smplx body pose
    """
    color_red = np.array([255, 0, 0, 224], dtype=np.uint8)
    color_blue = np.array([0, 0, 255, 224], dtype=np.uint8)
    color_gray = np.array([32, 32, 32, 128], dtype=np.uint8)
    
    scene = pyrender.Scene()

    scene_KDtree = KDTree(scene_mesh.vertices, leaf_size=int(0.8 * len(scene_mesh.vertices)))
    scene_mesh.visual.vertex_colors = color_gray
    atten_score = (atten_score - atten_score.min()) / (atten_score.max() - atten_score.min())

    verts_score = np.zeros(len(scene_mesh.vertices))
    for i, pos in enumerate(atten_pos):
        dist, indic = scene_KDtree.query(pos.reshape(1, -1), k=len(scene_mesh.vertices), return_distance=True, sort_results=True)
        indic = indic[0]
        verts_score[indic] += atten_score[i] * (1 - np.sqrt(dist[0] / dist[0].max()))
    
    verts_score = (verts_score - verts_score.min()) / (verts_score.max() - verts_score.min())
    coe_color = get_multi_colors_by_hsl(color_blue[0:3], color_red[0:3], verts_score)
    scene_mesh.visual.vertex_colors = coe_color

    ## other meshes
    if cam_pose is not None:
        camera_pose = cam_pose
    else:
        ## use fixed top-down view
        scene_center = scene_mesh.vertices.mean(axis=0)
        camera_pose = np.eye(4)
        camera_pose[0:3, -1] = np.array([*scene_center[0:2], 7])

    camera = pyrender.camera.IntrinsicsCamera(
        fx=1060, fy=1060,
        cx=951.30, cy=536.77)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    axis_mesh = trimesh.creation.axis(origin_size=0.02)
    scene.add(pyrender.Mesh.from_trimesh(axis_mesh, smooth=False), 'mesh_axis')
    scene.add(pyrender.Mesh.from_trimesh(scene_mesh, smooth=False), 'mesh_scene')
    if pred_target_object is not None:
        target_center = trimesh.creation.uv_sphere(radius=0.3)
        target_center.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
        target_center.apply_translation(pred_target_object)
        scene.add(pyrender.Mesh.from_trimesh(target_center, smooth=False), 'target_center')

    r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
    color, _ = r.render(scene)
    color = color.astype(np.float32) / 255.0
    img = Image.fromarray((color * 255).astype(np.uint8))
    os.makedirs(save_folder, exist_ok = True)
    img.save(os.path.join(save_folder, 'attention.png'))
    r.delete()

def render_scene_body_meshes(
    save_path: str,
    scene_mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
    camera_pose: np.ndarray,
):
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    body_m = pyrender.Mesh.from_trimesh(body_mesh, material=material)
    scene_m = pyrender.Mesh.from_trimesh(scene_mesh)

    scene = pyrender.Scene()
    W = 512
    H = 512
    camera = pyrender.camera.IntrinsicsCamera(
        fx=550, fy=550,
        cx=340, cy=256)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(scene_m, 'scene')
    scene.add(body_m, 'body')

    r = pyrender.OffscreenRenderer(viewport_width=W,viewport_height=H)
    color, _ = r.render(scene)
    color = color.astype(np.float32) / 255.0
    img = Image.fromarray((color * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    img.save(save_path)
    r.delete()

def frame2video(path, video, start, end=-1, framerate=30):
    """ Convert images frames to video
    Args:
        path: the image frame path with format string
        video: save path of the video result
        start: start frame index
        end: end frame index
        framerate: fps of the output video
    """
    if end == -1:
        cmd = 'ffmpeg -y -framerate {} -start_number {} -i "{}" -pix_fmt yuv420p "{}"'.format(framerate, start, path, video)
    else:
        cmd = 'ffmpeg -y -framerate {} -start_number {} -i "{}" -vframes {} -pix_fmt yuv420p -vcodec h264 "{}"'.format(framerate, start, path, end-start+1, video)
    os.system(cmd)

def frame2gif(frames, gif, duration=33.33):
    """ Convert image frames to gif, use PIL to implement the convertion.
    Args:
        frames: a image list or a image directory
        gif: save path of gif result
        duration: the duration(ms) of images in gif
    """
    if isinstance(frames, list):
        frames = natsorted(frames)
    elif os.path.isdir(frames):
        frames = natsorted(glob.glob(os.path.join(frames, '*.png')))
    else:
        raise Exception('Unsupported input type.')

    img, *imgs = [Image.open(f) for f in frames]

    os.makedirs(os.path.dirname(gif), exist_ok=True)
    img.save(fp=gif, format='GIF', append_images=imgs,
            save_all=True, duration=duration, loop=0)

