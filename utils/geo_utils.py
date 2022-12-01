import numpy as np
import trimesh
from pyquaternion import Quaternion as Q

def normal(ps=None, p1=None, p2=None, p3=None, unit=False):
    """ Compute normal vector of a triangular face
    Args:
        ps: the vertices array of the face with shape <3, 3>
        p1, p2, p3: three discrete points forming the triangular face
        unit: a bool type, normalize the normal vector to unit or not
    
    Return:
        the normal vector with shape <3>
    """
    if ps is not None:
        p1, p2, p3 = ps
    
    assert p1 is not None, 'Invalid Input.'

    v1 = p2 - p1
    v2 = p3 - p1
    n  = np.cross(v1, v2)

    if unit:
        return n / np.linalg.norm(n) + 1e-9
    else:
        return n

def is_point_in_cuboid(cuboid_verts, p):
    """ Judge whether a 3D space point is in a cuboid

    The vertices indication is as following:
      3 _ _ _ _7
      /|      /|
    1/_|_ _ _/5|
    | 2|_ _ _|_|6
    | /      | /
   0|/_ _ _ _|/4

    Args:
        cuboid_verts: the corner vertices <8, 3> of the cuboid
        p: the point coordinate <3>
    
    Return:
        a bool value
    """
    v0, v1, v2, v3, v4, v5, v6, v7 = cuboid_verts

    n1 = normal(p1=v0, p2=v1, p3=v2)
    n2 = normal(p1=v4, p2=v5, p3=v6)
    vec1 = p - v0
    vec2 = p - v4
    if np.dot(n1, vec1) * np.dot(n2, vec2) > 0:
        return False
    
    n1 = normal(p1=v0, p2=v1, p3=v4)
    n2 = normal(p1=v2, p2=v3, p3=v6)
    vec1 = p - v0
    vec2 = p - v2
    if np.dot(n1, vec1) * np.dot(n2, vec2) > 0:
        return False
    
    n1 = normal(p1=v0, p2=v2, p3=v4)
    n2 = normal(p1=v1, p2=v3, p3=v5)
    vec1 = p - v0
    vec2 = p - v1
    if np.dot(n1, vec1) * np.dot(n2, vec2) > 0:
        return False
    
    return True

def make_M_from_tqs(t, q, s):
    q = np.array(q)
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = Q(q).rotation_matrix
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M

def compute_rotate_matrix(norm_before, norm_after):
    """ Compute rotate matrix according two vector
    Args:
        norm_before: vector before rotation
        norm_after: vector after rotation
    
    Return:
        Rotation matrix with shape <3, 3>.
    """
    e1 = norm_before
    e2 = norm_after

    na = e1[1] * e2[2] - e1[2] * e2[1]
    nb = -(e1[0] * e2[2] - e1[2] * e2[0])
    nc = e1[0] * e2[1] - e1[1] * e2[0]

    cos_angle = norm_before.dot(norm_after) / (np.linalg.norm(norm_before) * np.linalg.norm(norm_after))
    sin_angle = np.sqrt(1 - cos_angle ** 2)

    w = np.array([na, nb, nc])
    w = w / np.linalg.norm(w)

    w = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    
    # mat = np.eye(3) + w * np.sin(angle) + np.matmul(w, w) * (1 - np.cos(angle))
    mat = np.eye(3) + w * sin_angle + np.matmul(w, w) * (1 - cos_angle)

    m = np.eye(4)
    m[0:3, 0:3] = mat

    return m

def create_unit_bbox():
    verts = np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ], dtype=np.float32)
    color = np.array([[255, 0, 0, 255]] * 8, dtype=np.uint8)
    return trimesh.PointCloud(vertices=verts, colors=color)

def create_vector(orient_vec: np.ndarray=None):
    vec = trimesh.load('./vector.ply')
    mat = compute_rotate_matrix(np.array([1, 0, 0]), orient_vec)
    vec.apply_transform(mat)
    vec.visual.vertex_colors[:, 0:3] = np.array([255, 0, 0], dtype=np.uint8)
    return vec

def rotate_2D_points_along_z_axis(points: np.ndarray, angle: float):
    """ Rotate 2D points along z axis

    Args:
        points: 2D points coordinates on xy plane
        angle: rotation angle
    
    Return:
        Rotated points on xy plane
    """
    r = Q(axis=[0, 0, 1], angle=angle).rotation_matrix[0:2, 0:2]
    return (r @ points.T).T

import torch 
# import torch_scatter

def smplx_signed_distance(object_points, smplx_vertices, smplx_face):
    """ Compute signed distance between query points and mesh vertices.
    
    Args:
        object_points: (B, O, 3) query points in the mesh.
        smplx_vertices: (B, H, 3) mesh vertices.
        smplx_face: (F, 3) mesh faces.
    
    Return:
        signed_distance_to_human: (B, O) signed distance to human vertex on each object vertex
        closest_human_points: (B, O, 3) closest human vertex to each object vertex
        signed_distance_to_obj: (B, H) signed distance to object vertex on each human vertex
        closest_obj_points: (B, H, 3) closest object vertex to each human vertex
    """
    # compute vertex normals
    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
    e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
    smplx_face_normal = torch.cross(e1, e2)     # (B, F, 3)

    # compute vertex normal
    smplx_vertex_normals = torch.zeros(smplx_vertices.shape).float().cuda()
    smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1)

    # compute paired distance of each query point to each face of the mesh
    pairwise_distance = torch.norm(object_points.unsqueeze(2) - smplx_vertices.unsqueeze(1), dim=-1, p=2)    # (B, O, H)
    
    # find the closest face for each query point
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)  # (B, O)
    closest_human_point = smplx_vertices.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)
    query_to_surface = closest_human_point - object_points
    query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)
    closest_vertex_normals = smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human    # (B, O)
    
    # find signed distance to object from human
    # signed_distance_to_object = torch.zeros([pairwise_distance.shape[0], pairwise_distance.shape[2]]).float().cuda()-10  # (B, H)
    # signed_distance_to_object, closest_obj_points_idx = torch_scatter.scatter_max(signed_distance_to_human, closest_human_points_idx, out=signed_distance_to_object)
    # closest_obj_points_idx[closest_obj_points_idx == pairwise_distance.shape[1]] = 0
    # closest_object_point = object_points.gather(1, closest_obj_points_idx.unsqueeze(-1).repeat(1,1,3))
    # return signed_distance_to_human, closest_human_point, signed_distance_to_object, closest_object_point, smplx_vertex_normals
    return signed_distance_to_human, closest_human_point
