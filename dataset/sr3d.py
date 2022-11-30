from collections import defaultdict
import pandas as pd
import random
import re


class SR3D_Util():

    def __init__(self, sr3d_csv: str):
        self.utterance = {}

        df = pd.read_csv(sr3d_csv)
        for i in range(df.shape[0]):
            scan_id = df.loc[i]['scan_id'] 
            target_id = df.loc[i]['target_id']
            utterance = df.loc[i]['utterance']
            refer_type = df.loc[i]['coarse_reference_type']

            if scan_id not in self.utterance:
                self.utterance[scan_id] = defaultdict(list)
            
            self.utterance[scan_id][target_id].append({
                'utterance': utterance,
                'refer_type': refer_type,
            })
    
    def contain_utterance(self, scene_id, object_id):
        if scene_id not in self.utterance:
            return False
        
        if object_id not in self.utterance[scene_id]:
            return False

        return True
    
    def generate_utterance(self, action, scene_id, object_id, object_label, cnt):
        action_phrase = self.get_action_phrase(action)

        if self.contain_utterance(scene_id, object_id):
            utterance_list = self.utterance[scene_id][object_id]

            utterance_dict = defaultdict(list)
            utterance_cnt = defaultdict(int)
            for item in utterance_list:
                u = item['utterance']
                t = item['refer_type']
                utterance_dict[t].append(u)
                utterance_cnt[t] += 1

            index = {}            
            for key in utterance_dict:
                index[key] = 0
                random.shuffle(utterance_dict[key])
            
            keys = list(utterance_dict.keys())
            res_u = []
            for i in range(cnt):
                r_t = random.choice(keys)

                u = utterance_dict[r_t][index[r_t]]
                res_u.append(self.replace_utterance_with_action(u, action_phrase))

                index[r_t] = 1 + index[r_t]
                index[r_t] = index[r_t] % utterance_cnt[r_t]
            
            return res_u
        else:
            template = self.get_utterance_template(action)
            template = template.replace('%action_phrase%', action_phrase)
            template = template.replace('%target%', object_label)

            return [template] * cnt
    
    def get_utterance_template(self, action):
        if action == 'sit':
            return "%action_phrase% the %target%"
        elif action == 'stand up':
            return "%action_phrase% the %target%"
        elif action == 'walk':
            return "%action_phrase% the %target%"
        elif action == 'lie':
            return "%action_phrase% the %target%"
        elif action == 'jump':
            return "%action_phrase% the %target%"
        elif action == 'turn':
            return "%action_phrase% the %target%"
        elif action == 'place something':
            return "%action_phrase% the %target%"
        elif action == 'open something':
            return "%action_phrase% the %target%"
        elif action == 'knock':
            return "%action_phrase% the %target%"
        elif action == 'dance':
            return "%action_phrase% the %target%"
        else:
            raise Exception('Unsupported action.')

    def replace_utterance_with_action(self, utterance, action_phrase):
        pattern = r'\bfind\b|\bchoose\b|\bselect\b|\bpick\b'
        u, n = re.subn(pattern, action_phrase, utterance)
        if n == 0:
            return '{} {}'.format(action_phrase, u)
        else:
            return u
    
    def get_action_phrase(self, action):
        if action == 'sit':
            action_phrase = 'sit on'
        elif action == 'stand up':
            action_phrase = 'stand up from'
        elif action == 'walk':
            action_phrase = 'walk to'
        elif action == 'lie':
            action_phrase = 'lie on'
        elif action == 'jump':
            action_phrase = 'jump near'
        elif action == 'turn':
            action_phrase = 'turn to'
        elif action == 'place something':
            action_phrase = 'place something on'
        elif action == 'open something':
            action_phrase = 'open'
        elif action == 'knock':
            action_phrase = 'knock on'
        elif action == 'dance':
            action_phrase = 'dance near'
        else:
            raise Exception('Unsupported action.')
        
        return action_phrase
        