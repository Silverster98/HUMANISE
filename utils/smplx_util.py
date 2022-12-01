from typing import List, Tuple
import numpy as np
from pyquaternion import Quaternion as Q
import smplx
import torch

left_foot = {"verts_ind": [5774, 5781, 5789, 5790, 5791, 5792, 5793, 5794, 5797, 5805, 5806, 5807, 5808, 5813, 5814, 5815, 5816, 5817, 5818, 5824, 5827, 5830, 5831, 5832, 5839, 5840, 5842, 5843, 5844, 5847, 5850, 5851, 5854, 5855, 5859, 5861, 5862, 5864, 5865, 5869, 5902, 5906, 5907, 5908, 5909, 5910, 5911, 5912, 5913, 5914, 5915, 5916, 5917, 8866, 8867, 8868, 8879, 8880, 8881, 8882, 8883, 8884, 8888, 8889, 8890, 8891, 8897, 8898, 8899, 8900, 8901, 8902, 8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8929, 8930, 8934], "faces_ind": [4257, 4259, 4274, 4283, 4284, 4286, 4288, 4307, 4309, 4311, 4314, 4320, 4340, 4354, 4371, 4422, 4462, 4463, 4464, 4467, 4469, 4470, 4471, 4472, 4478, 4530, 4545, 4546, 4547, 4549, 4550, 4551, 4552, 4553, 4554, 4568, 4581, 4616, 4618, 4874, 4875, 4876, 4978, 5004, 5072, 5074, 5075, 5076, 5077, 5204, 5263, 5264, 5265, 5268, 5324, 5353, 5375, 5376, 5386, 5390, 5627, 5684, 5694, 5712, 5783, 5785, 5860, 5867, 14790, 14805, 14808, 14814, 14815, 14817, 14819, 14834, 14840, 14845, 14858, 14859, 14871, 14922, 14953, 14993, 14994, 14995, 14998, 15000, 15001, 15002, 15003, 15009, 15061, 15076, 15077, 15078, 15080, 15081, 15082, 15083, 15084, 15085, 15099, 15112, 15147, 15149, 15405, 15406, 15407, 15434, 15509, 15535, 15602, 15604, 15605, 15606, 15607, 15733, 15793, 15794, 15797, 15853, 15904, 15905, 15915, 15919, 16156, 16213, 16240, 16311, 16313, 16388, 16394, 16395, 16415]}

right_foot = {"verts_ind": [8468, 8484, 8485, 8487, 8488, 8499, 8500, 8501, 8502, 8507, 8508, 8509, 8510, 8511, 8512, 8521, 8522, 8523, 8524, 8525, 8526, 8527, 8530, 8533, 8534, 8536, 8537, 8538, 8541, 8543, 8544, 8545, 8546, 8548, 8549, 8555, 8556, 8558, 8559, 8563, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8611, 8654, 8655, 8656, 8667, 8668, 8669, 8670, 8671, 8672, 8676, 8677, 8678, 8679, 8685, 8686, 8687, 8688, 8689, 8690, 8691, 8692, 8693, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8705, 8706, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716], "faces_ind": [8100, 8105, 8107, 8108, 8112, 8113, 8117, 8122, 8123, 8132, 8133, 8139, 8149, 8157, 8171, 8182, 8188, 8192, 8227, 8229, 8230, 8231, 8232, 8295, 8296, 8297, 8298, 8299, 8300, 8301, 8302, 8303, 8304, 8305, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315, 8316, 8317, 8318, 8319, 8320, 8323, 8324, 8325, 8326, 8327, 8328, 8329, 8330, 8331, 8333, 8334, 8335, 8336, 8337, 8558, 8559, 8560, 8561, 8562, 18623, 18628, 18630, 18631, 18635, 18636, 18640, 18645, 18646, 18648, 18656, 18662, 18666, 18672, 18680, 18694, 18703, 18705, 18711, 18750, 18752, 18753, 18754, 18755, 18818, 18819, 18820, 18821, 18822, 18823, 18824, 18825, 18826, 18827, 18828, 18829, 18830, 18831, 18832, 18833, 18834, 18835, 18836, 18837, 18838, 18839, 18840, 18841, 18842, 18843, 18847, 18848, 18849, 18850, 18851, 18852, 18853, 18854, 18856, 18857, 18858, 18859, 18860, 19081, 19082, 19083, 19084]}

gluteus = {"faces_ind": [1573, 3081, 3125, 3483, 4047, 4049, 4086, 4189, 4190, 4194, 4198, 4199, 4213, 4215, 4237, 4293, 4402, 4511, 4592, 4615, 4745, 4851, 4856, 4859, 4912, 4929, 5124, 5276, 5313, 5339, 5343, 5354, 5414, 5428, 5600, 5605, 5625, 5685, 5775, 5781, 5843, 5874, 6007, 6008, 6009, 6022, 6173, 7899, 7902, 7903, 7908, 7909, 7912, 7914, 7915, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924, 7925, 7926, 7927, 7928, 8012, 8013, 8014, 8015, 8016, 8050, 8061, 8430, 8431, 8441, 8442, 8448, 8521, 8522, 8523, 8524, 8525, 8526, 8539, 8564, 8606, 8732, 12122, 13615, 13659, 13916, 14015, 14579, 14581, 14618, 14721, 14722, 14726, 14730, 14731, 14744, 14746, 14768, 14824, 14933, 15042, 15123, 15146, 15276, 15382, 15390, 15443, 15460, 15653, 15805, 15842, 15868, 15872, 15883, 15943, 15957, 16129, 16134, 16154, 16214, 16303, 16309, 16371, 16402, 16535, 16537, 16550, 16701, 18426, 18427, 18432, 18433, 18436, 18438, 18439, 18440, 18441, 18442, 18443, 18444, 18445, 18446, 18447, 18448, 18449, 18450, 18451, 18452, 18536, 18537, 18538, 18539, 18540, 18573, 18584, 18585, 18952, 18953, 18963, 18964, 18970, 19043, 19044, 19045, 19046, 19047, 19048, 19061, 19086, 19128, 19254], "verts_ind": [3462, 3463, 3464, 3465, 3466, 3468, 3469, 3470, 3471, 3472, 3473, 3483, 3501, 3512, 3513, 3514, 3515, 3867, 3884, 3885, 5574, 5575, 5596, 5613, 5614, 5659, 5661, 5665, 5666, 5667, 5673, 5674, 5675, 5676, 5678, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5693, 5694, 5695, 5696, 5712, 5713, 5714, 5715, 5934, 6223, 6224, 6225, 6226, 6227, 6229, 6230, 6231, 6232, 6233, 6234, 6273, 6274, 6275, 6276, 6634, 6635, 6651, 7145, 8353, 8354, 8355, 8359, 8360, 8361, 8362, 8363, 8365, 8366, 8367, 8368, 8369, 8370, 8372, 8374, 8375, 8376, 8377, 8378, 8379, 8380, 8381, 8382, 8383, 8384, 8385, 8386, 8387, 8388, 8389, 8390, 8405, 8406, 8407, 8408, 8409]}

butt = {"verts_ind": [3462, 3465, 3468, 3469, 3513, 3514, 4065, 4066, 5557, 5574, 5596, 5673, 5665, 5681, 5693, 5713, 5714, 5931, 6226, 6223, 6229, 6230, 6274, 6275, 8359, 8367, 8375, 8387, 8407, 8408]}

left_knee = {"verts_ind": [3648, 3662, 3673, 3674, 3676]}

right_knee = {"verts_ind": [6409, 6423, 6434, 6435, 6437]}

right_hand = {"verts_ind": [7488]}

JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

markers = {
    "gender": "unknown",
    "markersets": [
        {
            "distance_from_skin": 0.0095,
            "indices": {
                "C7": 3832,
                "CLAV": 5533,
                "LANK": 5882,
                "LFWT": 3486,
                "LBAK": 3336,
                "LBCEP": 4029,
                "LBSH": 4137,
                "LBUM": 5694,
                "LBUST": 3228,
                "LCHEECK": 2081,
                "LELB": 4302,
                "LELBIN": 4363,
                "LFIN": 4788,
                "LFRM2": 4379,
                "LFTHI": 3504,
                "LFTHIIN": 3998,
                "LHEE": 8846,
                "LIWR": 4726,
                "LKNE": 3682,
                "LKNI": 3688,
                "LMT1": 5890,
                "LMT5": 5901,
                "LNWST": 3260,
                "LOWR": 4722,
                "LBWT": 5697,
                "LRSTBEEF": 5838,
                "LSHO": 4481,
                "LTHI": 4088,
                "LTHMB": 4839,
                "LTIB": 3745,
                "LTOE": 5787,
                "MBLLY": 5942,
                "RANK": 8576,
                "RFWT": 6248,
                "RBAK": 6127,
                "RBCEP": 6776,
                "RBSH": 7192,
                "RBUM": 8388,
                "RBUSTLO": 8157,
                "RCHEECK": 8786,
                "RELB": 7040,
                "RELBIN": 7099,
                "RFIN": 7524,
                "RFRM2": 7115,
                "RFRM2IN": 7303,
                "RFTHI": 6265,
                "RFTHIIN": 6746,
                "RHEE": 8634,
                "RKNE": 6443,
                "RKNI": 6449,
                "RMT1": 8584,
                "RMT5": 8595,
                "RNWST": 6023,
                "ROWR": 7458,
                "RBWT": 8391,
                "RRSTBEEF": 8532,
                "RSHO": 6627,
                "RTHI": 6832,
                "RTHMB": 7575,
                "RTIB": 6503,
                "RTOE": 8481,
                "STRN": 5531,
                "T8": 5487,
                "LFHD": 707,
                "LBHD": 2026,
                "RFHD": 2198,
                "RBHD": 3066
            },
            "marker_radius": 0.0095,
            "type": "body"
        }
    ]
}

marker_indic = list(markers['markersets'][0]['indices'].values())

class SMPLX_Util():

    @staticmethod
    def convert_smplx_verts_transfomation_matrix_to_body(T, trans, orient, pelvis):
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
    
    @staticmethod
    def get_body_vertices_sequence(smplx_folder: str, pkl: Tuple, num_betas=16):
        """ Get a sequence body from motion pkl annotation

        Args:
            smplx_folder: the smplx weights folder
            pkl: motion annotation
        
        Return:
            Body vertices sequence (numpy), body mesh face, joints sequence (numpy)
        """
        trans, orient, betas, body_pose, hand_pose = pkl
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
        joints = output.joints.detach().cpu().numpy()

        return vertices, body_model.faces, joints

    @staticmethod
    def get_foot_mid_trajectory_2d(body_vertex_sequence: np.ndarray, return_radius: bool = False):
        """ Get the trajectory of two foots center

        Args:
            body_vertex_sequence: the sequence of body vertex
            return_radius: return half distance of left foot and right foot
        
        Return:
            Left foot trajectory and right foot trajectory
        """
        left_foot_mid_xy = body_vertex_sequence[:, left_foot['verts_ind'], 0:2].mean(axis=1)
        right_foot_mid_xy = body_vertex_sequence[:, right_foot['verts_ind'], 0:2].mean(axis=1)
        traj_xy = (left_foot_mid_xy + right_foot_mid_xy) * 0.5

        if return_radius:
            r = np.sqrt(((left_foot_mid_xy - traj_xy) ** 2).sum(axis=-1))
            return traj_xy, r
        return traj_xy
    
    @staticmethod
    def get_foots_trajectory_2d(body_vertex_sequence: np.ndarray):
        """ Get the trajectory of two foots

        Args:
            body_vertex_sequence: the sequence of body vertex
        
        Return:
            Left foot trajectory and right foot trajectory
        """
        left_foot_traj = body_vertex_sequence[:, left_foot['verts_ind'], 0:2].mean(axis=1)
        right_foot_traj = body_vertex_sequence[:, right_foot['verts_ind'], 0:2].mean(axis=1)

        return left_foot_traj, right_foot_traj
    
    @staticmethod
    def get_foots_trajectory(body_vertex_sequence: np.ndarray):
        """ Get the trajectory of two foots in 3D space

        Args:
            body_vertex_sequence: the sequence of body vertex
        
        Return:
            Left foot trajectory and right foot trajectory
        """
        left_foot_traj = body_vertex_sequence[:, left_foot['verts_ind'], :].mean(axis=1)
        right_foot_traj = body_vertex_sequence[:, right_foot['verts_ind'], :].mean(axis=1)

        return left_foot_traj, right_foot_traj
    
    @staticmethod
    def get_butt_verts(body_vertex: np.ndarray):
        """ Get verts of butt

        Args:
            body_vertex: the body vertex array
        
        Return:
            
        """
        verts = body_vertex[butt['verts_ind'], :]
        return verts
    
    @staticmethod
    def get_knee_verts(body_vertex: np.ndarray):
        """ Get verts of knees

        Args:
            body_vertex: the body vertex array
        
        Return::
            Left knee vertices, Right knee vertices
        """
        left_knee_verts = body_vertex[left_knee['verts_ind'], :]
        right_knee_verts = body_vertex[right_knee['verts_ind'], :]
        return left_knee_verts, right_knee_verts
    
    @staticmethod
    def get_right_hand(body_vertex: np.ndarray):
        """ Get right hand position

        Args:
            body_vertex: the body vertex array
        
        Return::
            Right hand position
        """
        right_hand_verts = body_vertex[right_hand['verts_ind'], :]
        return right_hand_verts
    
    # @staticmethod
    # def get_body_orient(body_vertex: np.ndarray):
    #     """ Get the orientation of body

    #     Args:
    #         body_vertex: body vertices
        
    #     Return:
    #         The orient vector
    #     """
    #     c = body_vertex[5939]
    #     v1 = body_vertex[6717] - c
    #     v2 = body_vertex[4292] - c

    #     v3 = body_vertex[4400] - c
    #     v4 = body_vertex[5940] - c
    #     n1 = np.cross(v1, v2)
    #     n2 = np.cross(v3, v4)
    #     m_n = 0.5 * (n1 + n2)
    #     return m_n / np.linalg.norm(m_n)
    
    @staticmethod
    def get_body_orient(body_joints: np.ndarray):
        """ Get the orientation of body

        Args:
            body_vertex: body vertices
        
        Return:
            The orient vector
        """
        left_shoulder = body_joints[JOINT_NAMES.index('left_shoulder')]
        right_shoulder = body_joints[JOINT_NAMES.index('right_shoulder')]
        left_hip = body_joints[JOINT_NAMES.index('left_hip')]
        right_hip = body_joints[JOINT_NAMES.index('right_hip')]

        v1 = left_shoulder - left_hip
        v2 = right_shoulder - left_hip
        n1 = np.cross(v1, v2)

        v3 = left_shoulder - right_hip
        v4 = right_shoulder - right_hip
        n2 = np.cross(v3, v4)

        m_n = 0.5 * (n1 + n2)
        return m_n / np.linalg.norm(m_n)

    @staticmethod
    def get_sparse_skeleton(joints_all: np.ndarray):
        """ Return sparse skeleton joints

        joints = [   
            'pelvis',
            'left_hip',
            'right_hip',
            'spine1',
            'left_knee',
            'right_knee',
            'spine2',
            'left_ankle',
            'right_ankle',
            'spine3',
            'left_foot',
            'right_foot',
            'neck',
            'left_collar',
            'right_collar',
            'head',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist'
        ]
        """

        if len(joints_all.shape) == 2:
            return joints_all[0:21, :]
        
        if len(joints_all.shape) == 3:
            return joints_all[:, 0:21, :]

        raise Exception('Unexcepted joints input.')
    
    @staticmethod
    def get_stem_skeleton(joints_all: np.ndarray):
        """ Return sparse skeleton joints

        joints = [   
            'pelvis', 0
            'left_hip', 1
            'right_hip', 2
            'spine1', 3
            'left_knee', 4
            'right_knee', 5
            'spine2', 6
            'left_ankle', 7
            'right_ankle', 8
            'spine3', 9
            'neck', 12
            'left_collar', 13
            'right_collar', 14
            'head', 15
        ]
        """
        indics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15]

        if len(joints_all.shape) == 2:
            return joints_all[indics, :]
        
        if len(joints_all.shape) == 3:
            return joints_all[:, indics, :]

        raise Exception('Unexcepted joints input.')
