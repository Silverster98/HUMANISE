from typing import Tuple
from pandas.core.common import flatten
import re

def get_labels(ann, file):
    # Get sequence labels and frame labels if they exist, with duration
    seq_l, frame_l = None, None
    if 'extra' not in file:
        if ann['seq_ann'] is not None:
            seq_l = [seg['raw_label'] for seg in ann['seq_ann']['labels']]
        if ann['frame_ann'] is not None:
            frame_l = [(seg['raw_label'], seg['start_t'], seg['end_t']) for seg in ann['frame_ann']['labels']]
    else:
        # Load labels from 1st annotator (random) if there are multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = [seg['raw_label'] for seg in ann['seq_anns'][0]['labels']]
        if ann['frame_anns'] is not None:
            frame_l = [(seg['raw_label'], seg['start_t'], seg['end_t']) for seg in ann['frame_anns'][0]['labels']]
    return seq_l, frame_l

def get_cats(ann, file):
    # Get sequence labels and frame labels if they exist, without duration
    seq_l, frame_l = [], []
    if 'extra' not in file:
        if ann['seq_ann'] is not None:
            seq_l = flatten([seg['act_cat'] for seg in ann['seq_ann']['labels']])
        if ann['frame_ann'] is not None:
            frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = flatten([seg['act_cat'] for seq_ann in ann['seq_anns'] for seg in seq_ann['labels']])
        if ann['frame_anns'] is not None:            
            frame_l = flatten([seg['act_cat'] for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']])
            
    return list(seq_l), list(frame_l)

def get_subset(babel_data: dict, dataset: str=None, action: str=None, duration: float=None):
    """ Get sub dataset of babel, split by dataset name or action or duration

    Args:
        babel_dataset: the babel dataset annotation dict
        dataset: split by dataset, provide the dataset name
        action : split by action, provide the action keywords
        durantion: split by duration, provide the duration
    
    Return:
        New sub dataset (a dict).
    """
    def contains_action(seq_l, frame_l, action):
        if seq_l is not None:
            for s_l in seq_l:
                if action in s_l:
                    return True
        if frame_l is not None:
            for f_l, _, _ in frame_l:
                if action in f_l:
                    return True
        return False

    subset = {}
    for key in babel_data.keys():
        seq_l, frame_l = get_labels(babel_data[key], '')

        if dataset is not None:
            if dataset not in babel_data[key]['feat_p']:
                continue
    
        if action is not None:
            if not contains_action(seq_l, frame_l, action):
                continue
        
        if duration is not None:
            ## TO DO
            pass

        subset[key] = babel_data[key]

    return subset

def get_motion_segment(motion: dict, action: str):
    """ Extract motion segment from a sequece of motion

    Args:
        motion: a motion annotation dict
        action: action type
    
    Return:
        A list of all satisfied segments
    """
    def action_in_act_cat(action, act_cat):
        for act in act_cat:
            if re.search(r'\b'+action+r'\b', act):
                return True
        return False

    segment = []
    if motion['frame_ann'] is not None:
        for seg in motion['frame_ann']['labels']:
            if action_in_act_cat(action, seg['act_cat']):
                segment.append({
                    'seg_id': seg['seg_id'], 
                    'babel_sid': motion['babel_sid'],
                    'feat_p': motion['feat_p'],
                    'action_label': action,
                    'raw_label': seg['raw_label'],
                    'start_t': seg['start_t'],
                    'end_t': seg['end_t'],
                    'url': motion['url'],
                    'dur': motion['dur']
                })
    
    if len(segment) == 0:
        if motion['seq_ann'] is None:
            return segment
        
        for seg in motion['seq_ann']['labels']:
            if seg['act_cat'] is None:
                continue
            
            if action_in_act_cat(action, seg['act_cat']):
                segment.append({
                    'seg_id': seg['seg_id'],
                    'babel_sid': motion['babel_sid'],
                    'feat_p': motion['feat_p'],
                    'action_label': action,
                    'raw_label': seg['raw_label'],
                    'start_t': 0,
                    'end_t': motion['dur'],
                    'url': motion['url'],
                    'dur': motion['dur']
                })
                break
    
    return segment

def convert_smplh_path_to_smplx_neutral_path(feat_p: str):
    """ Convert the path of smplh annotation to the path of smplx neutral annotation

    Args:
        feat_p: original smplh feat path
    
    Return:
        Converted smplx neutral feat path
    """
    tmp_d = feat_p.split('/')[0:1] + feat_p.split('/')[2:]
    smplx_neutral_feat_p = '/'.join(tmp_d).replace(' ', '_').replace('_poses.', '_stageii.')

    return smplx_neutral_feat_p
