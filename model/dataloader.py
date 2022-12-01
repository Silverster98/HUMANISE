import glob
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import utils.configuration as config
from utils.smplx_util import SMPLX_Util
from pyquaternion import Quaternion as Q
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
from utils.utilities import Console
from utils.model_utils import GeometryTransformer
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
# import smplx
from trimesh import transform_points
from natsort import natsorted

############################################################
## Pose Dataset calss
############################################################
class MotionDataset(Dataset):
    """ Motion Dataset for fetching a padded motion sequence data
    """

    def __init__(self, phase: str, npoints: int=8192, use_color: bool=False, use_normal: bool=False, max_lang_len: int=128, max_motion_len: int=120, action: str='*', num_betas: int=10, num_pca_comps: int=12):
        """ Init function, prepare train/val data
        """
        self.anno_folder = config.align_data_folder
        self.scene_folder = config.preprocess_scene_folder
        self.motion_folder = config.pure_motion_folder
        self.phase = phase
        self.npoints = npoints
        self.use_color = use_color
        self.use_normal = use_normal
        self.max_lang_len = max_lang_len
        self.max_motion_len = max_motion_len
        self.num_betas = num_betas
        self.num_pca_comps = num_pca_comps

        aligns = natsorted(glob.glob(os.path.join(self.anno_folder, '{}/*/anno.pkl'.format(action))))
        if phase == 'train':
            motion_data = self._get_train_anno_split(aligns)
        elif phase == 'val':
            motion_data = self._get_val_anno_split(aligns)
        else:
            raise Exception('Unexpected phase.')
        
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # vp, ps = load_model(config.vposer_folder, model_code=VPoser,
        #                     remove_words_in_model_weights='vp_model.',
        #                     disable_grad=True)
        # self.vposer = vp.to('cpu') # use cpu

        ## prepare motion data
        self._prepare_motion_data_list(motion_data)
        Console.log('Total {} motion data examples in {} set.'.format(len(self.motion_data_list), self.phase))
        
    def _get_train_anno_split(self, align_annos: list):
        self.anno_list = []
        self.scene_data = {}
        
        motion_data = {}

        for a in align_annos:
            with open(a, 'rb') as fp:
                data = pickle.load(fp)
            
            for i, p in enumerate(data):
                action = p['action']
                scene_id = p['scene']
                motion_id = p['motion']

                if int(scene_id[5:9]) < 600:
                    self.anno_list.append(p)

                    if scene_id not in self.scene_data:
                        scene_data = np.load(os.path.join(self.scene_folder, scene_id+'.npy'))
                        self.scene_data[scene_id] = scene_data.astype(np.float32)
                    
                    if motion_id not in motion_data:
                        with open(os.path.join(self.motion_folder, action, motion_id, 'motion.pkl'), 'rb') as fp:
                            mdata = pickle.load(fp)
                        motion_data[motion_id] = mdata
        
        return motion_data
    
    def _get_val_anno_split(self, align_annos: list):
        self.anno_list = []
        self.scene_data = {}
        
        motion_data = {}

        for a in align_annos:
            with open(a, 'rb') as fp:
                data = pickle.load(fp)
            
            for i, p in enumerate(data):
                action = p['action']
                scene_id = p['scene']
                motion_id = p['motion']

                if int(scene_id[5:9]) >= 600:
                    self.anno_list.append(p)

                    if scene_id not in self.scene_data:
                        scene_data = np.load(os.path.join(self.scene_folder, scene_id+'.npy'))
                        self.scene_data[scene_id] = scene_data.astype(np.float32)
                    
                    if motion_id not in motion_data:
                        with open(os.path.join(self.motion_folder, action, motion_id, 'motion.pkl'), 'rb') as fp:
                            mdata = pickle.load(fp)
                        motion_data[motion_id] = mdata
        
        return motion_data
    
    def _get_anchor_frame_index(self, action: str):
        if action == 'sit':
            return -1
        elif action == 'stand up':
            return 0
        elif action == 'walk':
            return -1
        elif action == 'lie':
            return -1
        else:
            raise Exception('Unexcepted action type.')
    
    def _prepare_motion_data_list(self, motion_data):
        """ Prepare motion data by constructing the motion data according to annotation data

        Args:
            motion_data: a dict, all pure motion data, `{motion_id: motion_data, ...}`
        """
        self.motion_data_list = []

        for anno_index in tqdm(range(len(self.anno_list))):
            anno_data = self.anno_list[anno_index]
            motion_id = anno_data['motion']
            motion_trans = anno_data['translation']
            motion_rotat = anno_data['rotation']
            anchor_frame_index = self._get_anchor_frame_index(anno_data['action'])

            gender, origin_trans, origin_orient, betas, pose_body, pose_hand, pose_jaw, pose_eye, joints_traj = motion_data[motion_id]
            ## transform smplx bodies in motion sequence, convert smplx parameter, be careful
            cur_trans, cur_orient, cur_pelvis = self._transform_smplx_from_origin_to_sampled_position(
            motion_trans, motion_rotat, origin_trans, origin_orient, joints_traj[:, 0, :], anchor_frame_index)

            ### representation transfer
            ## 1. only use 10 components of betas
            betas = betas[:self.num_betas]

            mdata = (anno_index, cur_trans, cur_orient, betas.copy(), pose_body, pose_hand, cur_pelvis)
            self.motion_data_list.append(mdata)
    
    def _pad_utterance(self, token_id_seq, pad_val: int=0):
        """ Add padding to token id sequnece of utterance

        Args:
            token_id_seq: a sequence of token id
            pad_val: default is 0
        
        Return:
            Padded token id sequence
        """
        if len(token_id_seq) > self.max_lang_len:
            return token_id_seq[0:self.max_lang_len]
        else:
            return token_id_seq + [pad_val] * (self.max_lang_len - len(token_id_seq))

    def _transform_smplx_from_origin_to_sampled_position(
        self,
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

    def _pad_motion(self, trans: np.ndarray, orient: np.ndarray, pose_body: np.ndarray, pose_hand: np.ndarray):
        """ Add padding to smplx parameter sequence

        Args:
            trans:
            orient:
            pose_body:
            pose_hand:
        
        Return:
            Padded smplx parameters, i.e. trans, orient, pose_body, pose_hand, and a mask array
        """
        if trans.shape[0] > self.max_motion_len:
            trans = trans[0:self.max_motion_len]
            orient = orient[0:self.max_motion_len]
            pose_body = pose_body[0:self.max_motion_len]
            pose_hand = pose_hand[0:self.max_motion_len]
        
        S, D = trans.shape
        trans_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        trans = np.concatenate([trans, trans_padding], axis=0)

        _, D = orient.shape
        orient_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        orient = np.concatenate([orient, orient_padding], axis=0)

        _, D = pose_body.shape
        pose_body_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        pose_body = np.concatenate([pose_body, pose_body_padding], axis=0)

        _, D = pose_hand.shape
        pose_hand_padding = np.zeros((self.max_motion_len - S, D), dtype=np.float32)
        pose_hand = np.concatenate([pose_hand, pose_hand_padding], axis=0)

        ## generate mask
        motion_mask = np.zeros(self.max_motion_len, dtype=bool)
        motion_mask[S:] = True

        return trans, orient, pose_body, pose_hand, motion_mask

    def __getitem__(self, index):
        ## get motion data, e.g. smplx parameter sequence
        ## smplx parameters are pre-processed, need to add padding
        anno_index, trans, orient, betas, pose_body, pose_hand, pelvis = self.motion_data_list[index]
        anno_data = self.anno_list[anno_index]
        
        ## process scene
        scene_id = anno_data['scene']
        scene_trans = anno_data['scene_translation']

        scene_data = self.scene_data[scene_id]
        if len(scene_data) < self.npoints:
            sel_indic = np.random.choice(len(scene_data), self.npoints, replace=True)
        else:
            sel_indic = np.random.choice(len(scene_data), self.npoints, replace=False)
        scene_data = scene_data[sel_indic] # sample npoints points in scene point cloud
        scene_data[:, 0:3] += scene_trans # translate scene to origin center

        point_set = scene_data[:, :3]

        if self.use_color:
            rgb = scene_data[:, 3:6] / 255
            point_set = np.concatenate([point_set, rgb], axis=1)
        
        if self.use_normal:
            normal = scene_data[:, 6:9]
            point_set = np.concatenate([point_set, normal], axis=1)


        ## process utterance
        utterance = anno_data['utterance']

        lang_desc_str = "[CLS] " + utterance + " [SEP]" # add CLS and SEP
        tokenized_desc = self.tokenizer.tokenize(lang_desc_str) # tokenize
        token_id = self.tokenizer.convert_tokens_to_ids(tokenized_desc) # convert to id
        lang_desc = self._pad_utterance(token_id, pad_val=0) # padding
        lang_desc = np.array(lang_desc, dtype=np.long)
        lang_mask = [float(i == 0) for i in lang_desc] # mask
        lang_mask = np.array(lang_mask, dtype=np.bool)
        

        ## prepare scene transformation matrix
        scene_T = np.eye(4, dtype=np.float32)
        scene_T[0:3, -1] = scene_trans

        ## data augmentation
        if self.phase == 'train':
            rand_T, cur_scene, trans, orient, pelvis = self._augment(point_set[:, 0:3], trans, orient, pelvis)
            point_set[:, 0:3] = cur_scene
            scene_T = rand_T @ scene_T
        
        target_object_mask = scene_data[:, -2] == anno_data['object_id'] # target object mask
        target_object_verts  = point_set[target_object_mask, 0:3] # target object verts
        target_object_center = (target_object_verts.min(axis=0) + target_object_verts.max(axis=0)) * 0.5 # target object bounding box center
        orient_6D = GeometryTransformer.convert_to_6D_rot(torch.tensor(orient)).numpy()
        trans = trans.astype(np.float32)

        ## padding motion sequence
        trans, orient_6D, pose_body, pose_hand, motion_mask = self._pad_motion(trans, orient_6D, pose_body, pose_hand)

        # print(scene_id)
        # print(scene_T.dtype, scene_T.shape)
        # print(point_set.dtype, point_set.shape)
        # print(lang_desc.dtype, lang_desc.shape, lang_desc)
        # print(lang_mask.dtype, lang_mask.shape, lang_mask)
        # print(trans.dtype, trans.shape)
        # print(orient.dtype, orient.shape)
        # print(betas.dtype, betas.shape)
        # print(pose_body.dtype, pose_body.shape)
        # print(pose_hand.dtype, pose_hand.shape)
        # print(motion_mask.dtype, motion_mask.shape)
        # exit(0)

        return scene_id, scene_T, point_set, utterance, lang_desc, lang_mask, trans, orient_6D, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask

    def __len__(self):
        return len(self.motion_data_list)
    
    def _augment(self, origin_scene, origin_trans, origin_orient, origin_pelvis):
        """ Random augmentation, transform the scene and motion with a random rotation angle and translation vector

        Args:
            origin_scene: origin scene vertices coordinates
            origin_trans: origin trans parameter of smplx
            origin_orient: origin orient parameter of smplx
            origin_pelvis: origin pelvis of smplx body
        
        Return:
            Random transformed scene vertices coordinates, trans parameters, orient parameters and smplx body pelvis
        """
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_trans = np.random.uniform(-2.0, 2.0, 2)

        T = Q(axis=np.array([0, 0, 1]), angle=random_angle).transformation_matrix
        T[0:2, -1] = random_trans

        cur_scene = transform_points(origin_scene, T)

        cur_trans = []
        cur_orient = []
        for i in range(len(origin_trans)):
            t, o = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(
                T, 
                origin_trans[i], 
                origin_orient[i], 
                origin_pelvis[i],
            )
            cur_trans.append(t)
            cur_orient.append(o)

        cur_trans = np.array(cur_trans)
        cur_orient = np.array(cur_orient)
        cur_pelvis = transform_points(origin_pelvis, T)

        return T, cur_scene, cur_trans, cur_orient, cur_pelvis


def collate_random_motion(data):
    (scene_id, scene_T, point_set, utterance, lang_desc, lang_mask,
    trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask) = zip(*data)


    ## convert to tensor
    lang_desc = torch.LongTensor(np.array(lang_desc))
    lang_mask = torch.BoolTensor(np.array(lang_mask))

    trans = torch.FloatTensor(np.array(trans))
    orient = torch.FloatTensor(np.array(orient))
    betas = torch.FloatTensor(np.array(betas))
    pose_body = torch.FloatTensor(np.array(pose_body))
    pose_hand = torch.FloatTensor(np.array(pose_hand))
    motion_mask = torch.BoolTensor(np.array(motion_mask))
    target_object_center = torch.FloatTensor(np.array(target_object_center))
    target_object_mask = torch.BoolTensor(np.array(target_object_mask))
    
    offset, count = [], 0
    for item in point_set:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    ## split points to coords and feats
    point_set = np.concatenate(point_set)
    point_set = torch.FloatTensor(np.array(point_set))
    point_coords = point_set[:, :3]
    point_feats = point_set[:, 3:]

    # print(scene_id)
    # print(scene_T)
    # print(point_coords.dtype, point_coords.shape)
    # print(point_feats.dtype, point_feats.shape)
    # print(utterance)
    # print(lang_desc.dtype, lang_desc.shape)
    # print(lang_mask.dtype, lang_mask.shape)
    # print(trans.dtype, trans.shape)
    # print(orient.dtype, orient.shape)
    # print(betas.dtype, betas.shape)
    # print(pose_body.dtype, pose_body.shape)
    # print(pose_hand.dtype, pose_hand.shape)
    # print(motion_mask.dtype, motion_mask.shape)
    # exit(0)

    batch = (
        scene_id,
        scene_T, 
        point_coords.cuda(),
        point_feats.cuda(),
        offset.cuda(),
        utterance,
        lang_desc.cuda(),
        lang_mask.cuda(),
        trans.cuda(),
        orient.cuda(),
        betas.cuda(),
        pose_body.cuda(),
        pose_hand.cuda(),
        motion_mask.cuda(),
        target_object_center.cuda(),
        target_object_mask.cuda(),
    )

    return batch