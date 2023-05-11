from collections import defaultdict
import os
from typing import Any, Tuple
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.basemodel import CondNet, MotionModel
from utils.utilities import Console, Ploter
from utils.visualization import render_attention, frame2video, render_reconstructed_motion_in_scene, render_sample_k_motion_in_scene
import utils.configuration as config
import trimesh
from utils.model_utils import GeometryTransformer
# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
from utils.smplx_util import SMPLX_Util, marker_indic
import smplx
from utils.geo_utils import smplx_signed_distance
import pickle

EPOCH_REPORT_TEMPLATE = """
----------------------summary----------------------
[train] train cost time:  {train_cost_time}
[train] train total loss: {train_total_loss}
[train] train ground loss: {train_ground_loss}
[train] train rec_h loss: {train_rec_h_loss}
[train] train KLdiv loss: {train_KLdiv_loss}
[train] train VPose loss: {train_VPose_loss}
[val]   val cost time:  {val_cost_time}
[val]   val total loss: {val_total_loss}
[val]   val ground loss: {val_ground_loss}
[val]   val rec_h loss: {val_rec_h_loss}
[val]   val KLdiv loss: {val_KLdiv_loss}
[val]   val VPose loss: {val_VPose_loss}
"""

BEST_REPORT_TEMPLATE = """
----------------------best----------------------
[best] best epoch: {best_epoch}
[best] best total loss: {best_total_loss}
[best] best ground loss: {best_ground_loss}
[best] best rec_h loss: {best_rec_h_loss}
[best] best KLdiv loss: {best_KLdiv_loss}
[best] best VPose loss: {best_VPose_loss}
"""

class MotionSolver():
    def __init__(self, conf: Any, dataloader: dict):
        self.config = conf

        self.cond_net = CondNet(self.config).to(self.config.device)
        self.base_model = MotionModel(self.config).to(self.config.device)
        self.dataloader = dataloader

        cond_scene_param = []
        cond_text_param = []
        cond_rest_param = []
        for name, param, in self.cond_net.named_parameters():
            if 'scene_model' in name:
                cond_scene_param.append(param)
            elif 'bert_model' in name:
                cond_text_param.append(param)
            else:
                cond_rest_param.append(param)
        self.optimizer_h = optim.Adam(
            [
                {'params': cond_scene_param, 'lr': self.config.lr * 0.1},
                {'params': cond_rest_param},
                {'params': list(self.base_model.parameters())},
            ],
            lr = self.config.lr
        )

        # log
        # contains all necessary info for all phases
        self.log = {phase: {} for phase in ["train", "val"]}
        self.dump_keys = ['loss', 'rec_loss', 'kl_loss', 'vposer_loss', 'rec_trans_loss', 'rec_orient_loss', 'rec_body_pose_loss', 'rec_body_mesh_loss', 'ground_loss', 'classify_loss']

        self.best = {
            'epoch': 0,
            'loss': float("inf"),
            'rec_loss': float("inf"),
            'kl_loss': float("inf"),
            'vposer_loss': float("inf"),
            'ground_loss': float("inf"),
            'classify_loss': float("inf"),
        }

        # report model size
        self._report_model_size()

        # vp, ps = load_model(config.vposer_folder, model_code=VPoser,
        #                     remove_words_in_model_weights='vp_model.',
        #                     disable_grad=True)
        # self.vposer = vp.to(self.config.device)
        self.smplx_model = smplx.create(config.smplx_folder, model_type='smplx',
            gender='neutral', ext='npz',
            num_betas=self.config.num_betas,
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
            batch_size=self.config.batch_size * self.config.max_motion_len,
        ).to(self.config.device)
    
    def _report_model_size(self):
        sum_scene = sum([param.nelement() for param in self.cond_net.scene_model.parameters()])
        sum_text = sum([param.nelement() for param in self.cond_net.bert_model.parameters()])
        sum_cond_net = sum([param.nelement() for param in self.cond_net.parameters()])
        sum_cond_rest = sum_cond_net - sum_text - sum_scene
        
        sum_base_model = sum([param.nelement() for param in self.base_model.parameters()])
        sum_encoder = sum([param.nelement() for param in self.base_model.encoder.parameters()])
        sum_decoder = sum([param.nelement() for param in self.base_model.decoder.parameters()])

        Console.log(
            'cond_scene ({}) + cond_text ({}) + cond_rest ({}) = condnet ({}), cvae parameters : {} = encoder ({}) + decoder ({})'.format(
                sum_scene, sum_text, sum_cond_rest, sum_cond_net, sum_base_model, sum_encoder, sum_decoder,
            )
        )
        Console.log('all parameters: {}'.format(sum_cond_net + sum_base_model))
    
    def _save_state_dict(self, epoch: int, name: str):
        saved_cond_net_state_dict = {k: v for k, v in self.cond_net.state_dict().items() if 'bert' not in k} # don't save bert weights
        torch.save({
            'epoch': epoch + 1,
            'cond_net_state_dict': saved_cond_net_state_dict,
            'base_model_state_dict': self.base_model.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict()
        }, os.path.join(self.config.log_dir, '{}.pth'.format(name)))

    def _load_state_dict(self):
        if os.path.isdir(self.config.resume_model):
            ckp_file = os.path.join(self.config.resume_model, 'model_last.pth')
        elif os.path.isfile(self.config.resume_model):
            ckp_file = self.config.resume_model
        else:
            return 0
        
        state_dict = torch.load(ckp_file)
        ## load cond net weight
        cond_net_dict = self.cond_net.state_dict()
        cond_net_dict.update(state_dict['cond_net_state_dict'])
        self.cond_net.load_state_dict(cond_net_dict)
        ## load cvae model weight
        self.base_model.load_state_dict(state_dict['base_model_state_dict'])

        Console.log('Load checkpoint: {}'.format(ckp_file))
        return state_dict['epoch']
    
    def _set_phase(self, phase: str):
        if phase == "train":
            self.cond_net.train()
            self.base_model.train()
        elif phase == "val":
            self.cond_net.eval()
            self.base_model.eval()
        else:
            raise Exception("Invalid phase")

    def _get_body_vertices(self, trans, orient, betas, body_pose, hand_pose):
        """ Get body vertices for regress body vertices

        Args:
            smplx paramters

        Returns:
            body vertices
        """
        torch_param = {}
        torch_param['body_pose'] = body_pose
        torch_param['betas'] = betas
        torch_param['transl'] = trans
        torch_param['global_orient'] = orient
        torch_param['left_hand_pose'] = hand_pose[:, 0:45]
        torch_param['right_hand_pose'] = hand_pose[:, 45:]

        output = self.smplx_model(return_verts=True, **torch_param)
        vertices = output.vertices
        joints = output.joints

        return vertices, joints

    def _cal_loss(self, x, rec_x, mu, logvar, motion_mask):
        """ Compute loss, rec_loss is `l1_loss`

        Args:
            x: ground truth, (trans, orient, body_pose)
            rec_x: reconstructed result, (trans, orient, body_pose)
            mu: 
            logvar:
            motion_mask:
        
        Return:
            recontruct loss and kl loss
        """
        target_object, action_category, trans, orient, body_pose, betas, hand_pose = x
        pred_target_object, action_logits, trans_rec, orient_rec, body_pose_rec, betas_rec, hand_pose_rec = rec_x

        B, S, _ = trans.shape
        
        ## ground loss
        ground_loss = F.mse_loss(target_object, pred_target_object)

        ## classify loss
        classify_loss = F.cross_entropy(action_logits, action_category)

        ## rec loss, smplx parameters
        rec_trans_loss = (F.l1_loss(
            trans, trans_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>
        rec_orient_loss = (F.l1_loss(
            orient, orient_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>
        rec_body_pose_loss = (F.l1_loss(
            body_pose, body_pose_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>

        ## rec loss, body vertices
        ## 1. get ground truth body vertices
        orient = GeometryTransformer.convert_to_3D_rot(orient.reshape(-1, 6))
        verts_gt, _ = self._get_body_vertices(
            torch.zeros(B * S, trans.shape[-1]).cuda(),
            torch.zeros(B * S, orient.shape[-1]).cuda(),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            body_pose.reshape(B * S, -1),
            torch.zeros(B * S, hand_pose.shape[-1]).cuda()
        )
        ## 2. get rec body vertices
        orient_rec = GeometryTransformer.convert_to_3D_rot(orient_rec.reshape(-1, 6))
        verts_rec, _ = self._get_body_vertices(
            torch.zeros(B * S, trans.shape[-1]).cuda(),
            torch.zeros(B * S, orient.shape[-1]).cuda(),
            betas_rec.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            body_pose_rec.reshape(B * S, -1),
            torch.zeros(B * S, hand_pose.shape[-1]).cuda()
        )
        ## 3. cal loss
        if self.config.all_body_vertices:
            verts_gt = verts_gt.reshape(B, S, -1)
            verts_rec = verts_rec.reshape(B, S, -1)
        else:
            verts_gt = verts_gt[:, marker_indic,:]
            verts_gt = verts_gt.reshape(B, S, -1)
            verts_rec = verts_rec[:, marker_indic,:]
            verts_rec = verts_rec.reshape(B, S, -1)

        rec_body_mesh_loss = (F.l1_loss(
            verts_gt, verts_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum()

        ## kl loss
        kl_loss = torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)

        return ground_loss, classify_loss, rec_trans_loss, rec_orient_loss, rec_body_pose_loss, rec_body_mesh_loss, kl_loss

    def _forward(
        self, 
        point_coords: torch.Tensor, 
        point_feats: torch.Tensor, 
        offset: torch.Tensor,
        lang_desc: torch.Tensor, 
        lang_mask: torch.Tensor, 
        betas: torch.Tensor,
        seq_mask: torch.Tensor,
        *args: Tuple,
    ):
        """ Forward function to predict

        Args:

            args: smplx parameters group for traning, (trans, orient, pose_body, pose_hand<optional>)

        Return:
            reconstruct results and (mu, logvar) pairs
        """

        D = [0]
        for arg in args:
            D.append(arg.shape[2])
        D = [sum(D[0:i]) for i in range(1, len(D)+1)]

        cond_latent, pred_target_object, classify_logits = self.cond_net(
            (point_coords, point_feats, offset),
            (lang_desc, lang_mask),
            betas,
        )

        rec_x, mu, logvar  = self.base_model(
            torch.cat(args, dim=-1),
            cond_latent,
            seq_mask,
        )
        rec_x = rec_x.permute(1, 0, 2)

        rec_args = []
        for i in range(len(args)):
            rec_args.append(rec_x[:, :, D[i]: D[i+1]])

        return pred_target_object, classify_logits, *rec_args, mu, logvar

    def _sample(
        self, 
        point_coords: torch.Tensor, 
        point_feats: torch.Tensor, 
        offset: torch.Tensor,
        lang_desc: torch.Tensor, 
        lang_mask: torch.Tensor, 
        betas: torch.Tensor,
        k: int=1,
        sample_type: str='fixed',
        nframe: int=60,
        *args: Tuple,
    ):
        """ Forward function to predict

        Args:

            args: smplx parameters group for traning, (trans, orient, pose_body, pose_hand<optional>)

        Return:
            reconstruct results and (mu, logvar) pairs
        """

        D = [0]
        for arg in args:
            D.append(arg.shape[2])
        D = [sum(D[0:i]) for i in range(1, len(D)+1)]

        cond_latent, pred_target_object, classify_logits, atten_score, scene_xyz = self.cond_net(
            (point_coords, point_feats, offset),
            (lang_desc, lang_mask),
            betas,
            need_atten_score = True,
        )

        sample_x, sample_m  = self.base_model.sample(
            cond_latent,
            k=k,
            nframe_type=sample_type,
            nframe=nframe,
        )
        sample_x = sample_x.permute(2, 1, 0, 3) # <B, K, S, D>

        rec_args = []
        for i in range(len(args)):
            rec_args.append(sample_x[:, :, :, D[i]: D[i+1]])

        return pred_target_object, classify_logits, *rec_args, sample_m, atten_score, scene_xyz

    def _train(self, train_dataloader: DataLoader, epoch_id: int):
        phase = 'train'
        self.log[phase][epoch_id] = defaultdict(list)
        
        for data in tqdm(train_dataloader):
            start = time.time()

            ## unpack data
            [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
            trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

            ## forward
            [pred_target_object_center, classify_logits, rec_trans, rec_orient, rec_pose_body, mu, logvar] = self._forward(
                point_coords, point_feats, offset, lang_desc, lang_mask, betas, motion_mask, trans, orient, pose_body,
            )
            [ground_loss, classify_loss, rec_trans_loss, rec_orient_loss, rec_body_pose_loss, rec_body_mesh_loss, kl_loss] = self._cal_loss(
                (target_object_center, action_category, trans, orient, pose_body, betas, pose_hand),
                (pred_target_object_center, classify_logits, rec_trans, rec_orient, rec_pose_body, betas, pose_hand),
                mu, logvar, motion_mask
            )
            
            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec * rec_orient_loss + \
                        self.config.weight_loss_rec_pose * rec_body_pose_loss + \
                        self.config.weight_loss_rec_vertex * rec_body_mesh_loss
            loss = self.config.weight_loss_classify * classify_loss + \
                    self.config.weight_loss_ground * ground_loss + \
                    self.config.weight_loss_kl * kl_loss + \
                    rec_loss

            ## backward
            self.optimizer_h.zero_grad()
            loss.backward()
            self.optimizer_h.step()

            ## record log
            iter_time = time.time() - start
            self.log[phase][epoch_id]['iter_time'].append(iter_time)
            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['ground_loss'].append(ground_loss.item())
            self.log[phase][epoch_id]['classify_loss'].append(classify_loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(rec_orient_loss.item())
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_body_mesh_loss'].append(rec_body_mesh_loss.item())
            self.log[phase][epoch_id]['kl_loss'].append(kl_loss.item())
            self.log[phase][epoch_id]['vposer_loss'].append(0.0)
            
    def _val(self, val_dataloader: DataLoader, epoch_id: int):
        phase = 'val'
        self.log[phase][epoch_id] = defaultdict(list)
        
        for data in val_dataloader:
            start = time.time()

            ## unpack data
            [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
            trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

            ## forward
            [predict_target_object_center, action_logits, rec_trans, rec_orient, rec_pose_body, mu, logvar] = self._forward(
                point_coords, point_feats, offset, lang_desc, lang_mask, betas, motion_mask, trans, orient, pose_body
            )
            [ground_loss, classify_loss, rec_trans_loss, rec_orient_loss, rec_body_pose_loss, rec_body_mesh_loss, kl_loss] = self._cal_loss(
                (target_object_center, action_category, trans, orient, pose_body, betas, pose_hand),
                (predict_target_object_center, action_logits, rec_trans, rec_orient, rec_pose_body, betas, pose_hand),
                mu, logvar, motion_mask
            )

            rec_loss = self.config.weight_loss_rec * rec_trans_loss + \
                        self.config.weight_loss_rec * rec_orient_loss + \
                        self.config.weight_loss_rec_pose * rec_body_pose_loss + \
                        self.config.weight_loss_rec_vertex * rec_body_mesh_loss
            loss = self.config.weight_loss_classify * classify_loss + \
                    self.config.weight_loss_ground * ground_loss + \
                    self.config.weight_loss_kl * kl_loss + \
                    rec_loss

            ## record log
            iter_time = time.time() - start
            self.log[phase][epoch_id]['iter_time'].append(iter_time)
            self.log[phase][epoch_id]['loss'].append(loss.item())
            self.log[phase][epoch_id]['ground_loss'].append(ground_loss.item())
            self.log[phase][epoch_id]['classify_loss'].append(classify_loss.item())
            self.log[phase][epoch_id]['rec_loss'].append(rec_loss.item())
            self.log[phase][epoch_id]['rec_trans_loss'].append(rec_trans_loss.item())
            self.log[phase][epoch_id]['rec_orient_loss'].append(rec_orient_loss.item())
            self.log[phase][epoch_id]['rec_body_pose_loss'].append(rec_body_pose_loss.item())
            self.log[phase][epoch_id]['rec_body_mesh_loss'].append(rec_body_mesh_loss.item())
            self.log[phase][epoch_id]['kl_loss'].append(kl_loss.item())
            self.log[phase][epoch_id]['vposer_loss'].append(0.0)

        ## ckeck best
        cur_criterion = 'loss'
        cur_best = np.mean(self.log[phase][epoch_id][cur_criterion])
        if cur_best < self.best[cur_criterion]:
            for key in self.best:
                if key != 'epoch':
                    self.best[key] = np.mean(self.log[phase][epoch_id][key])
            self.best['epoch'] = epoch_id
            
            ## save best
            self._save_state_dict(epoch_id, 'model_best')
    
    def _visualize(self, save_folder, scene_id, scene_trans_mat, utterance, motion1, motion2=None):
        scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))
        static_scene = trimesh.load(scene_path, process=False)
        static_scene.apply_transform(scene_trans_mat)

        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'u.json'), 'w') as fp:
            json.dump({'utterance': utterance, 'scene_id': scene_id}, fp)
        
        pkl1 = self._convert_compute_smplx_to_render_smplx(motion1)

        if motion2 is not None:
            pkl2 = self._convert_compute_smplx_to_render_smplx(motion2)
        else:
            pkl2 = None

        render_reconstructed_motion_in_scene(
            smplx_folder=config.smplx_folder,
            save_folder=os.path.join(save_folder, 'rendering'),
            pkl_rec=pkl1,
            scene_mesh=static_scene,
            pkl_gt =pkl2,
            num_betas=self.config.num_betas
        )

        frame2video(
            path=os.path.join(save_folder, 'rendering/%03d.png'),
            video=os.path.join(save_folder, 'motion.mp4'),
            start=0,
        )
        os.system('ffmpeg -i {} -vf "fps=10,scale=640:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize {}'.format(
            os.path.join(save_folder, 'motion.mp4'),
            os.path.join(save_folder, 'motion.gif')
        ))

        os.system('rm -rf {}'.format(os.path.join(save_folder, 'rendering/')))
    
    def _visualize_k_sample(self, save_folder, scene_id, scene_trans_mat, utterance, k_motions, motion_gt=None):
        scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))
        static_scene = trimesh.load(scene_path, process=False)
        static_scene.apply_transform(scene_trans_mat)

        os.makedirs(save_folder, exist_ok=True)
        with open(os.path.join(save_folder, 'u.json'), 'w') as fp:
            json.dump({'utterance': utterance, 'scene_id': scene_id}, fp)

        trans, orient, betas, pose_body, pose_hand = k_motions
        K, S, _ = trans.shape

        k_pkl = self._convert_compute_smplx_to_render_smplx(
            (
                trans.reshape(K * S, -1),
                orient.reshape(K * S, -1),
                betas,
                pose_body.reshape(K * S, -1),
                pose_hand.reshape(K * S, -1)
            )
        )
        trans, orient, betas, pose_body, pose_hand = k_pkl

        if motion_gt is not None:
            pkl_gt = self._convert_compute_smplx_to_render_smplx(motion_gt)
        else:
            pkl_gt = None

        render_sample_k_motion_in_scene(
            smplx_folder=config.smplx_folder,
            save_folder=os.path.join(save_folder, 'rendering'),
            pkl_rec=(
                trans.reshape(K, S, -1),
                orient.reshape(K, S, -1),
                betas,
                pose_body.reshape(K, S, -1),
                pose_hand.reshape(K, S, -1),
            ),
            scene_mesh=static_scene,
            pkl_gt =pkl_gt,
            num_betas=self.config.num_betas
        )

        frame2video(
            path=os.path.join(save_folder, 'rendering/%03d.png'),
            video=os.path.join(save_folder, 'motion.mp4'),
            start=0,
        )
        os.system('ffmpeg -i {} -vf "fps=10,scale=640:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize {}'.format(
            os.path.join(save_folder, 'motion.mp4'),
            os.path.join(save_folder, 'motion.gif')
        ))

        os.system('rm -rf {}'.format(os.path.join(save_folder, 'rendering/')))
    
    def _visualize_atten(self, save_folder, scene_id, scene_trans_mat, scene_xyz, atten_score, pred_target_object):
        scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id, scene_id))
        static_scene = trimesh.load(scene_path, process=False)
        static_scene.apply_transform(scene_trans_mat)

        render_attention(
            save_folder=save_folder,
            scene_mesh=static_scene,
            atten_score=atten_score,
            atten_pos=scene_xyz,
            pred_target_object=pred_target_object,
        )

    def check_data(self):
        print('-' * 20, 'check data', '-' * 20)
        for data in self.dataloader['train']:
            [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
            trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

            for index in range(len(trans)):
                print('check', index, scene_id[index], utterance[index], lang_desc[index], lang_mask[index], action_category)
                ## visualize training data
                S = trimesh.Scene()
                
                verts = point_coords.reshape(self.config.batch_size, self.config.npoints, 3).cpu().numpy()[index, :, 0:3]
                color = np.array(point_feats.reshape(self.config.batch_size, self.config.npoints, 3).cpu().numpy()[index, :, 0:3] * 255, dtype=np.uint8)
                t_mask = target_object_mask.cpu().numpy()[index]
                color[t_mask, 0:3] = np.array([0, 255, 0], dtype=np.uint8)
                S.add_geometry(trimesh.PointCloud(vertices=verts, colors=color))

                body_verts, body_face, body_pelvis = SMPLX_Util.get_body_vertices_sequence(
                    config.smplx_folder,
                    self._convert_compute_smplx_to_render_smplx((
                        trans[index][~motion_mask[index]], 
                        orient[index][~motion_mask[index]], 
                        betas[index], 
                        pose_body[index][~motion_mask[index]], 
                        pose_hand[index][~motion_mask[index]]
                    )),
                    num_betas=self.config.num_betas
                )
                for i in range(0, len(body_verts), 10):
                    S.add_geometry(trimesh.Trimesh(
                        vertices=body_verts[i],
                        faces=body_face
                    ))
                S.add_geometry(trimesh.creation.axis(origin_size=0.05))
                target_center = trimesh.creation.uv_sphere(radius=0.2)
                target_center.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
                target_center.apply_translation(target_object_center[index].cpu().numpy())
                S.add_geometry(target_center)
                S.show()

                ## visualize origin mesh
                S = trimesh.Scene()
                scene_path = os.path.join(config.scannet_folder, '{}/{}_vh_clean_2.ply'.format(scene_id[index], scene_id[index]))
                static_scene = trimesh.load(scene_path, process=False)
                static_scene.apply_transform(scene_T[index])
                S.add_geometry(static_scene)
                for i in range(0, len(body_verts), 10):
                    S.add_geometry(trimesh.Trimesh(
                        vertices=body_verts[i],
                        faces=body_face
                    ))
                S.add_geometry(trimesh.creation.axis(origin_size=0.05))
                target_center = trimesh.creation.uv_sphere(radius=0.2)
                target_center.visual.vertex_colors = np.array([255, 0, 0, 255], dtype=np.uint8)
                target_center.apply_translation(target_object_center[index].cpu().numpy())
                S.add_geometry(target_center)
                S.show()
            
            exit(0)
    
    def test_rec(self, dataset='val'):
        self._set_phase('val')
        self.config.resume_model = os.path.join(self.config.log_dir, 'model_best.pth')
        self._load_state_dict()

        with torch.no_grad():
            for i, data in enumerate(self.dataloader['test']):
                [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
                trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

                [predict_target_object_center, action_logits, rec_trans, rec_orient, rec_pose_body, mu, logvar] = self._forward(
                    point_coords, point_feats, offset, lang_desc, lang_mask, betas, motion_mask, trans, orient, pose_body
                )

                trans = trans[0][~motion_mask[0]]
                orient = orient[0][~motion_mask[0]]
                pose_body = pose_body[0][~motion_mask[0]]
                pose_hand = pose_hand[0][~motion_mask[0]]

                rec_trans = rec_trans[0][~motion_mask[0]]
                rec_orient = rec_orient[0][~motion_mask[0]]
                rec_pose_body = rec_pose_body[0][~motion_mask[0]]

                self._visualize(
                    os.path.join(self.config.log_dir, '{}_visual_rec'.format(dataset), str(i)),
                    scene_id[0],
                    scene_T[0],
                    utterance[0],
                    (rec_trans, rec_orient, betas, rec_pose_body, pose_hand),
                    (trans, orient, betas, pose_body, pose_hand)
                )

                if i > 100:
                    break

    def test_sample(self, dataset='val', k: int=1, sample_type: str='fixed'):
        self._set_phase('val')
        self.config.resume_model = os.path.join(self.config.log_dir, 'model_best.pth')
        self._load_state_dict()

        with torch.no_grad():
            for i, data in enumerate(self.dataloader['test']):
                [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
                trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

                nframe = torch.sum(~motion_mask)
                [pred_target_object, action_logits, rec_trans, rec_orient, rec_pose_body, rec_mask, atten_score, scene_xyz] = self._sample(
                    point_coords, point_feats, offset, lang_desc, lang_mask, betas, k, sample_type, nframe, trans, orient, pose_body
                )

                trans = trans[0][~motion_mask[0]]
                orient = orient[0][~motion_mask[0]]
                pose_body = pose_body[0][~motion_mask[0]]
                pose_hand = pose_hand[0][~motion_mask[0]]

                if sample_type == 'fixed':
                    rec_trans = rec_trans[0, :, ~motion_mask[0], :]
                    rec_orient = rec_orient[0, :, ~motion_mask[0], :]
                    rec_pose_body = rec_pose_body[0, :, ~motion_mask[0], :]
                    rec_pose_hand = torch.zeros(k, *pose_hand.shape)

                    self._visualize_k_sample(
                        os.path.join(self.config.log_dir, '{}_visual_sample_{}_{}'.format(dataset, k, sample_type), str(i)),
                        scene_id[0],
                        scene_T[0],
                        utterance[0],
                        (rec_trans, rec_orient, betas, rec_pose_body, rec_pose_hand),
                        (trans, orient, betas, pose_body, pose_hand)
                    )
                elif sample_type == 'rand':
                    for j in range(k):
                        mask_j = ~rec_mask[0][j]

                        rec_trans_j = rec_trans[0][j][mask_j]
                        rec_orient_j = rec_orient[0][j][mask_j]
                        rec_pose_body_j = rec_pose_body[0][j][mask_j]
                        rec_pose_hand_j = torch.zeros(mask_j.sum(), pose_hand.shape[-1])

                        self._visualize(
                            os.path.join(self.config.log_dir, '{}_visual_sample_{}_{}'.format(dataset, k, sample_type), str(i), 'sample_' + str(j)),
                            scene_id[0],
                            scene_T[0],
                            utterance[0],
                            (rec_trans_j, rec_orient_j, betas, rec_pose_body_j, rec_pose_hand_j),
                            (trans, orient, betas, pose_body, pose_hand)
                        )
                else:
                    raise Exception('Unexcepted nframes type.')

                ## visualize attention
                lang_token_atten = atten_score[0, 128:].detach().cpu().numpy()

                self._visualize_atten(
                    os.path.join(self.config.log_dir, '{}_visual_sample_{}_{}'.format(dataset, k, sample_type), str(i)),
                    scene_id[0], 
                    scene_T[0], 
                    scene_xyz[0].detach().cpu().numpy(), 
                    lang_token_atten[0, 0:128],
                    pred_target_object[0].detach().cpu().numpy(),
                )
                if i >= 100:
                    break

    def save_k_sample(self, dataset='val', k: int=10):
        self._set_phase('val')
        self.config.resume_model = os.path.join(self.config.log_dir, 'model_best.pth')
        self._load_state_dict()

        with torch.no_grad():
            for i, data in enumerate(self.dataloader['test']):
                [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
                trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

                nframe = torch.sum(~motion_mask)
                [pred_target_object, action_logits, rec_trans, rec_orient, rec_pose_body, rec_mask, atten_score, scene_xyz] = self._sample(
                    point_coords, point_feats, offset, lang_desc, lang_mask, betas, k, 'fixed', nframe, trans, orient, pose_body
                )

                ## save gt
                trans = trans[0][~motion_mask[0]]
                orient = orient[0][~motion_mask[0]]
                pose_body = pose_body[0][~motion_mask[0]]
                pose_hand = pose_hand[0][~motion_mask[0]]

                self._visualize(
                    os.path.join(self.config.log_dir, '{}_visual_sample_save/{}/gt'.format(dataset, str(i))),
                    scene_id[0],
                    scene_T[0],
                    utterance[0],
                    (trans, orient, betas, pose_body, pose_hand),
                    None
                )

                ## save k sample
                rec_trans = rec_trans[0, :, ~motion_mask[0], :]
                rec_orient = rec_orient[0, :, ~motion_mask[0], :]
                rec_pose_body = rec_pose_body[0, :, ~motion_mask[0], :]
                rec_pose_hand = torch.zeros(k, *pose_hand.shape)
                for j in range(k):
                    self._visualize(
                        os.path.join(self.config.log_dir, '{}_visual_sample_save/{}/sample_{}'.format(dataset, str(i), str(j))),
                        scene_id[0],
                        scene_T[0],
                        utterance[0],
                        (rec_trans[j], rec_orient[j], betas, rec_pose_body[j], rec_pose_hand[j]),
                        None
                    )
                    p = os.path.join(self.config.log_dir, '{}_visual_sample_save/{}/sample_{}/param.pkl'.format(dataset, str(i), str(j)))
                    with open(p, 'wb') as fp:
                        pickle.dump(
                            (scene_id[0], scene_T[0], utterance[0], (rec_trans[j], rec_orient[j], betas, rec_pose_body[j], rec_pose_hand[j])
                        ), fp)

                ## visualize attention
                lang_token_atten = atten_score[0, 128:].detach().cpu().numpy()

                self._visualize_atten(
                    os.path.join(self.config.log_dir, '{}_visual_sample_save/{}'.format(dataset, str(i))),
                    scene_id[0], 
                    scene_T[0], 
                    scene_xyz[0].detach().cpu().numpy(), 
                    lang_token_atten[0, 0:128],
                    None,
                )
                
                if i >= 20:
                    break

    def _convert_compute_smplx_to_render_smplx(self, smplx_tensor_tuple):
        trans1, orient1, betas1, pose_body1, pose_hand1 = smplx_tensor_tuple
        trans1 = trans1.detach().cpu().numpy()
        orient1 = GeometryTransformer.convert_to_3D_rot(orient1.detach()).cpu().numpy()
        betas1 = betas1.detach().cpu().numpy()
        pose_body1 = pose_body1.detach().cpu().numpy()
        pose_hand1 = pose_hand1.detach().cpu().numpy()

        return (trans1, orient1, betas1, pose_body1, pose_hand1)

    def __call__(self):

        start_epoch = self._load_state_dict()
        
        for epoch_id in range(start_epoch, self.config.num_epoch):
            Console.log('epoch {:0>5d} starting...'.format(epoch_id))

            ## train
            self._set_phase('train')
            self._train(self.dataloader['train'], epoch_id)

            ## val
            with torch.no_grad():
                self._set_phase('val')
                self._val(self.dataloader['val'], epoch_id)

            ## report log
            self._epoch_report(epoch_id)
            self._dump_log(epoch_id)

        # print best
        self._best_report()

        # save model
        Console.log("saving last models...\n")
        self._save_state_dict(epoch_id, 'model_last')
    
    def _epoch_report(self, epoch_id: int):
        Console.log("epoch [{}/{}] done...".format(epoch_id+1, self.config.num_epoch))

        epoch_report_str = EPOCH_REPORT_TEMPLATE.format(
            train_cost_time=round(np.mean(self.log['train'][epoch_id]['iter_time']), 5),
            train_total_loss=round(np.mean(self.log['train'][epoch_id]['loss']), 5),
            train_ground_loss=round(np.mean(self.log['train'][epoch_id]['ground_loss']), 5),
            train_rec_h_loss=round(np.mean(self.log['train'][epoch_id]['rec_loss']), 5),
            train_KLdiv_loss=round(np.mean(self.log['train'][epoch_id]['kl_loss']), 5),
            train_VPose_loss=round(np.mean(self.log['train'][epoch_id]['vposer_loss']), 5),
            val_cost_time=round(np.mean(self.log['val'][epoch_id]['iter_time']), 5),
            val_total_loss=round(np.mean(self.log['val'][epoch_id]['loss']), 5),
            val_ground_loss=round(np.mean(self.log['val'][epoch_id]['ground_loss']), 5),
            val_rec_h_loss=round(np.mean(self.log['val'][epoch_id]['rec_loss']), 5),
            val_KLdiv_loss=round(np.mean(self.log['val'][epoch_id]['kl_loss']), 5),
            val_VPose_loss=round(np.mean(self.log['val'][epoch_id]['vposer_loss']), 5),
        )
        Console.log(epoch_report_str)
    
    def _best_report(self):
        Console.log("training completed...")

        best_report_str = BEST_REPORT_TEMPLATE.format(
            best_epoch=self.best['epoch'],
            best_total_loss=self.best['loss'],
            best_ground_loss=self.best['ground_loss'],
            best_rec_h_loss=self.best['rec_loss'],
            best_KLdiv_loss=self.best['kl_loss'],
            best_VPose_loss=self.best['vposer_loss'],
        )
        Console.log(best_report_str)

    def _dump_log(self, epoch_id: int):
        dump_logs = {}

        for key in self.dump_keys:
            k = 'train/' + key
            dump_logs[k] = {
                'plot': True,
                'step': epoch_id,
                'value': np.mean(self.log['train'][epoch_id][key])
            }
        for key in self.dump_keys:
            k = 'val/' + key
            dump_logs[k] = {
                'plot': True,
                'step': epoch_id,
                'value': np.mean(self.log['val'][epoch_id][key])
            }

        Ploter.write(dump_logs)
    
    def _compute_rec_error(self, x, rec_x, mu, logvar, motion_mask):
        target_object, trans, orient, body_pose, betas, hand_pose = x
        pred_target_object, trans_rec, orient_rec, body_pose_rec, _, _ = rec_x

        B, S, _ = trans.shape
        
        ## ground loss
        ground_error = F.mse_loss(target_object, pred_target_object)

        ## rec loss, smplx parameters
        rec_trans_error = (F.l1_loss(
            trans, trans_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>
        rec_orient_error = (F.l1_loss(
            orient, orient_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>
        rec_body_pose_error = (F.l1_loss(
            body_pose, body_pose_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, Seq_Len, D>.mean(-1) * <B, Seq_Len>

        ## rec loss, body vertices
        ## 1. get ground truth body vertices
        orient = GeometryTransformer.convert_to_3D_rot(orient.reshape(-1, 6))
        verts_gt, joints_gt = self._get_body_vertices(
            trans.reshape(B * S, -1),
            orient.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            body_pose.reshape(B * S, -1),
            hand_pose.reshape(B * S, -1)
        )
        ## 2. get rec body vertices
        orient_rec = GeometryTransformer.convert_to_3D_rot(orient_rec.reshape(-1, 6))
        verts_rec, joints_rec = self._get_body_vertices(
            trans_rec.reshape(B * S, -1),
            orient_rec.reshape(B * S, -1),
            betas.reshape(B, 1, -1).repeat(1, S, 1).reshape(B * S, -1),
            body_pose_rec.reshape(B * S, -1),
            hand_pose.reshape(B * S, -1),
        )
        ## 3. mpvpe
        verts_gt = verts_gt.reshape(B, S, -1, 3)
        verts_rec = verts_rec.reshape(B, S, -1, 3)
        # rec_vertex_error = (F.l1_loss(
        #     verts_gt, verts_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum()
        rec_vertex_error = (torch.sqrt(((verts_gt - verts_rec) ** 2).sum(-1)).mean(-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, S, N, 3>

        ## 4. mpjpe
        joints_gt = joints_gt.reshape(B, S, -1, 3)
        joints_rec = joints_rec.reshape(B, S, -1, 3)
        # rec_joints_error = (F.l1_loss(
        #     joints_gt, joints_rec, reduction='none').mean(dim=-1) * (~motion_mask)).sum() / (~motion_mask).sum()
        rec_joints_error = (torch.sqrt(((joints_gt - joints_rec) ** 2).sum(-1)).mean(-1) * (~motion_mask)).sum() / (~motion_mask).sum() # <B, S, N, 3>

        ## kl loss
        kl_error = torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)

        return ground_error, rec_trans_error, rec_orient_error, rec_body_pose_error, rec_vertex_error, rec_joints_error, kl_error
    
    def _report_rec_metric_results(self, write_file='./rec.json'):
        self._set_phase('val')
        self.config.resume_model = os.path.join(self.config.log_dir, 'model_best.pth')
        self._load_state_dict()

        with torch.no_grad():
            ground_error_all = []
            rec_trans_error_all = []
            rec_orient_error_all = []
            rec_body_pose_error_all = []
            rec_vertex_error_all = []
            rec_joints_error_all = []
            kl_error_all = []
            frames = []

            for i, data in enumerate(self.dataloader['test']):
                ## unpack data
                [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
                trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data

                ## forward
                [predict_target_object_center, action_logits, rec_trans, rec_orient, rec_pose_body, mu, logvar] = self._forward(
                    point_coords, point_feats, offset, lang_desc, lang_mask, betas, motion_mask, trans, orient, pose_body
                )

                [ground_error, rec_trans_error, rec_orient_error, rec_body_pose_error, rec_vertex_error, rec_joints_error, kl_error] = self._compute_rec_error(
                    (target_object_center, trans, orient, pose_body, betas, pose_hand),
                    (predict_target_object_center, rec_trans, rec_orient, rec_pose_body, betas, pose_hand),
                    mu, logvar, motion_mask
                )

                ground_error_all.append(ground_error.item())
                rec_trans_error_all.append(rec_trans_error.item())
                rec_orient_error_all.append(rec_orient_error.item())
                rec_body_pose_error_all.append(rec_body_pose_error.item())
                rec_vertex_error_all.append(rec_vertex_error.item())
                rec_joints_error_all.append(rec_joints_error.item())
                kl_error_all.append(kl_error.item())
                frames.append((~motion_mask).sum().item())
            
            results = {}
            results['ground_error'] = ground_error_all
            results['trans_error']  = rec_trans_error_all
            results['orient_error'] = rec_orient_error_all
            results['pose_error']   = rec_body_pose_error_all
            results['vertex_error'] = rec_vertex_error_all
            results['joints_error'] = rec_joints_error_all
            results['kl_error']     = kl_error_all
            results['frames']       = frames

            ## sequence level error
            results['sequence_level'] = {
                'ground error:': sum(ground_error_all) / len(ground_error_all),
                'rec trans error:': sum(rec_trans_error_all) / len(rec_trans_error_all),
                'rec orient error:': sum(rec_orient_error_all) / len(rec_orient_error_all),
                'rec pose error:': sum(rec_body_pose_error_all) / len(rec_body_pose_error_all),
                'rec vertex error:': sum(rec_vertex_error_all) / len(rec_vertex_error_all),
                'rec joints error:': sum(rec_joints_error_all) / len(rec_joints_error_all),
                'kl error:': sum(kl_error_all) / len(kl_error_all),
            }
            ## frame level error
            results['frame_level'] = {
                'ground error': (np.array(ground_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'rec trans error': (np.array(rec_trans_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'rec orient error': (np.array(rec_orient_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'rec pose error': (np.array(rec_body_pose_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'rec vertex error': (np.array(rec_vertex_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'rec joints error': (np.array(rec_joints_error_all) * np.array(frames)).sum() / np.array(frames).sum(),
                'kl error': sum(kl_error_all) / len(kl_error_all),
            }

            os.makedirs(os.path.dirname(write_file), exist_ok=True)
            with open(write_file, 'w') as fp:
                json.dump(results, fp)
    
    def _report_sample_metric_results(self, k=10, sample_type='fixed', write_file='./sample.json'):
        self._set_phase('val')
        self.config.resume_model = os.path.join(self.config.log_dir, 'model_best.pth')
        self._load_state_dict()

        with torch.no_grad():
            dist_to_obj = []
            non_collision_score = []
            contact_score = []
            frames = []

            for i, data in enumerate(self.dataloader['test']):
                [scene_id, scene_T, point_coords, point_feats, offset, utterance, lang_desc, lang_mask, 
                trans, orient, betas, pose_body, pose_hand, motion_mask, target_object_center, target_object_mask, action_category] = data
                anchor_index = 0 if action_category[0] == 2 else -1

                nframe = torch.sum(~motion_mask)
                [pred_target_object, action_logits, rec_trans, rec_orient, rec_pose_body, rec_mask, atten_score, scene_xyz] = self._sample(
                    point_coords, point_feats, offset, lang_desc, lang_mask, betas, k, sample_type, nframe, trans, orient, pose_body
                )

                ## distance to target
                dist_to_obj_case = []
                non_collision_score_case = []
                contact_score_case = []
                ## physical error
                for j in range(k):
                    # print(i, j)
                    mask_j = ~rec_mask[0][j]

                    rec_trans_j = rec_trans[0][j][mask_j]
                    rec_orient_j = rec_orient[0][j][mask_j]
                    rec_pose_body_j = rec_pose_body[0][j][mask_j]
                    rec_pose_hand_j = torch.zeros(mask_j.sum(), pose_hand.shape[-1])

                    pkl = self._convert_compute_smplx_to_render_smplx((
                        rec_trans_j, rec_orient_j, betas, rec_pose_body_j, rec_pose_hand_j))
                    body_vertices, body_faces, body_joints = SMPLX_Util.get_body_vertices_sequence(
                        config.smplx_folder, pkl, num_betas=self.config.num_betas)
                    body_vertices = torch.tensor(body_vertices).cuda()
                    body_faces = torch.tensor(body_faces.astype(np.int64)).cuda()
                    body_joints = torch.tensor(body_joints).cuda()
                    
                    ## distance to target, find the object object closest to body
                    object_verts = point_coords[target_object_mask[0], 0:3].unsqueeze(dim=0) # point_coords is <B * N, 3>, <B, O, 3>
                    anchor_body_verts = body_vertices[anchor_index, :, :].unsqueeze(dim=0) # <B, H, 3>
                    object_to_human_sdf, _ = smplx_signed_distance(
                        object_points=object_verts, smplx_vertices=anchor_body_verts, smplx_face=body_faces) # <B, O> = D(<B, O, 3>, <B, H, 3>)
                    d_o = min(object_to_human_sdf.max().item(), 0)
                    dist_to_obj_case.append(d_o)
                    

                    ## physical metric
                    non_collision_score_sequence = []
                    contact_score_sequence = []
                    scene_verts = point_coords.unsqueeze(dim=0)
                    for f in range(len(body_vertices)):
                        scene_to_human_sdf, _ = smplx_signed_distance(
                            object_points=scene_verts, smplx_vertices=body_vertices[f:f+1], smplx_face=body_faces)
                        # print(scene_to_human_sdf.shape)
                        sdf = scene_to_human_sdf.cpu().numpy() # <1, O>
                        non_collision_cur = np.sum(sdf <= 0) / sdf.shape[-1]
                        if np.sum(sdf > 0) > 0:
                            contact_cur = 1.0
                        else:
                            contact_cur = 0.0
                        
                        non_collision_score_sequence.append(non_collision_cur)
                        contact_score_sequence.append(contact_cur)
                    
                    non_collision_score_case.append(sum(non_collision_score_sequence) / len(non_collision_score_sequence))
                    contact_score_case.append(sum(contact_score_sequence) / len(contact_score_sequence))
                    
                dist_to_obj.append(sum(dist_to_obj_case) / len(dist_to_obj_case))
                non_collision_score.append(sum(non_collision_score_case) / len(non_collision_score_case))
                contact_score.append(sum(contact_score_case) / len(contact_score_case))
                frames.append(nframe.item())

            results = {}
            results['dist_to_obj'] = dist_to_obj
            results['non_collision_score'] = non_collision_score
            results['contact_score'] = contact_score
            results['frames'] = frames

            ## sequence level error
            results['sequence_level'] = {
                'dist_to_obj': sum(dist_to_obj) / len(dist_to_obj),
                'non_collision_score': sum(non_collision_score) / len(non_collision_score),
                'contact_score': sum(contact_score) / len(contact_score),
            }
            ## frame level error
            results['frame_level'] = {
                'dist_to_obj': (np.array(dist_to_obj) * np.array(frames)).sum() / np.array(frames).sum(),
                'non_collision_score': (np.array(non_collision_score) * np.array(frames)).sum() / np.array(frames).sum(),
                'contact_score': (np.array(contact_score) * np.array(frames)).sum() / np.array(frames).sum(),
            }

            os.makedirs(os.path.dirname(write_file), exist_ok=True)
            with open(write_file, 'w') as fp:
                json.dump(results, fp)
