import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

class ContinousRotReprDecoder(nn.Module):
    '''
    - this class encodes/decodes rotations with the 6D continuous representation
    - Zhou et al., On the continuity of rotation representations in neural networks
    - also used in the VPoser (see smplx)
    '''

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot

class GeometryTransformer():
    
    @staticmethod
    def verts_transform(verts_batch, cam_ext_batch):
        verts_batch_homo = F.pad(verts_batch, (0,1), mode='constant', value=1)
        verts_batch_homo_transformed = torch.matmul(verts_batch_homo,
                                                    cam_ext_batch.permute(0,2,1))

        verts_batch_transformed = verts_batch_homo_transformed[:,:,:-1]
        
        return verts_batch_transformed
    
    @staticmethod
    def convert_to_6D_rot(x_r):
        """ axis-angle to rotation matrix (6D)

        Args:
            x_r: <B, 3>

        Return:
            rotation matrix (6D), <B, 6>
        """

        xr_mat = ContinousRotReprDecoder.aa2matrot(x_r) # return [:,3,3]
        xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

        return xr_repr

    @staticmethod
    def convert_to_3D_rot(x_r):
        """ rotation matrix (6D) to axis-angle

        Args:
            x_r: <B, 6>
        
        Return:
            axis-angle, <B, 3>
        """

        xr_mat = ContinousRotReprDecoder.decode(x_r) # return [:,3,3]
        xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

        return xr_aa
    

