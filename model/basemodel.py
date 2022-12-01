import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from model.blocks import ResBlock, TransformerEncoderLayer, PositionalEncoding
from pytorch_pretrained_bert.modeling import BertModel
from utils.utilities import Console
from model.pointtransformer.pointtransformer_semseg import pointtransformer_feature_extractor as PointTransformerEnc
import random

class CondNet(nn.Module):
    def __init__(self, config):
        super(CondNet, self).__init__()
        self.config = config
        self.device = self.config.device

        self.pointnet2_cfg = {'is_msg': False, 'input_channels': 6, 'use_xyz': True, 'bn': True}
        self.co_attention_layer_nhead = 8
        self.scene_post_layer_out_size = 16

        ## scene model, point transformer
        fea_dim = 3 + int(self.config.use_color) * 3 + int(self.config.use_normal) * 3
        self.scene_model = PointTransformerEnc(c=fea_dim).cuda()
        if self.config.pretrained_scene_model != '':
            model_dict = torch.load(self.config.pretrained_scene_model)
            static_dict = self._process_scene_model_static_dict(model_dict)
            self.scene_model.load_state_dict(static_dict)
            Console.log('Load pre-trained scene model weigth \'{}\'..'.format(self.config.pretrained_scene_model))
        
        self.target_regressor = nn.Linear(self.config.condition_latent_size, 3)
        
        ## text model, bert
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.lang_linear = nn.Linear(self.config.lang_feat_size, self.config.scene_feat_size)

        ## cross attention
        self.co_attention_layer = TransformerEncoderLayer(
            self.config.scene_feat_size,
            self.co_attention_layer_nhead,
            batch_first=True,
        )

        ## post process modules
        self.scene_process_layers = nn.Sequential(
            nn.Linear(self.config.scene_feat_size + 3, self.scene_post_layer_out_size),
            nn.ReLU(),
        )
        self.scene_final_layer = nn.Linear(self.config.scene_group_size * self.scene_post_layer_out_size, self.config.scene_feat_size)

        ## final fusion layer
        self.fuse_layer = nn.Linear(
            self.config.scene_feat_size + self.config.scene_feat_size + self.config.body_feat_size,
            self.config.condition_latent_size,
        )

    def _process_scene_model_static_dict(self, model_dict):
        static_dict = {}
        for key in model_dict.keys():
            if 'enc' in key:
                static_dict[key] = model_dict[key]
        return static_dict

    def forward(self, scene, lang, body, need_atten_score: bool=False):
        B, _ = body.shape

        ## scene processing
        pxos = self.scene_model(scene)
        # scene_xyz.detach(), scene_feat.detach() ## don't fintune
        p5, x5, o5 = pxos[-1] # position <B * NS, 3>, feature <B * NS, 512>, offset <B> 

        _, NS, _ = p5.view(B, -1, 3).shape
        raw_fscene = x5.view(B, NS, -1) # <B, NS, 512>

        ## language processing
        word, word_mask = lang
        all_lang_encoder_layers, _ = self.bert_model(word, token_type_ids=None, attention_mask=word_mask)
        raw_fword = all_lang_encoder_layers[-1]
        raw_fword = raw_fword.detach() # <B, NL=32, C=768>, detach, don't fine-tune bert
        raw_fword = self.lang_linear(raw_fword) # <B, NL=32, C=512>
        
        ## co-attention
        scene_mask = torch.zeros(raw_fscene.shape[0:2], dtype=torch.bool, device=self.device)
        padding_mask_all = torch.cat([scene_mask, word_mask], dim=1).type(torch.bool)
        raw_fall = torch.cat([raw_fscene, raw_fword], dim=1) # <B, NS+NL, C=512>

        co_attention_output, atten_score = self.co_attention_layer(raw_fall, src_key_padding_mask=padding_mask_all) # <B, NS+NL, C=512>
        scene_hidden_states = co_attention_output[:, 0:NS, :] # <B, NS, C=512>
        lang_hidden_states = co_attention_output[:, NS:, :]

        ## post process
        ## scene, fuse all tokens
        scene_feature = torch.cat([scene_hidden_states, p5.view(B, -1, 3)], dim=-1) # <B, NS, C=512 + 3>
        scene_feature = self.scene_process_layers(scene_feature) # <B, NS, 16>
        scene_feature = self.scene_final_layer(scene_feature.view(B, -1)) # final_layer(<B, 16 * NS>) = <B, 512>
        ## language, use CLS token feature
        lang_feature = lang_hidden_states[:, 0, :]
        
        ## final fusion
        cond_feat = torch.cat([scene_feature, lang_feature, body], dim=1) # <B, 512 + 512 + 10>, concat conditions
        fusion_feat = self.fuse_layer(cond_feat) # <B, 512>


        ## for auxiliary task
        regress_center = self.target_regressor(fusion_feat)

        ## output
        if need_atten_score:
            return fusion_feat, regress_center, atten_score, p5.view(B, -1, 3)
        
        return fusion_feat, regress_center

class MotionEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, condition_latent_size: int, z_size: int=32):
        super(MotionEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_latent_size = condition_latent_size
        self.z_size = z_size

        self.sequential_ecoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size // 2,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=True,
        )
        self.res_module = nn.Sequential(
            ResBlock(self.hidden_size + self.condition_latent_size),
        )

        self.mu_encoder = nn.Linear(self.hidden_size + self.condition_latent_size, self.z_size) # mu
        self.logvar_encoder = nn.Linear(self.hidden_size + self.condition_latent_size, self.z_size) # logvar

    def forward(self, x, cond_z, seq_mask):
        B, nframe, f_dim = x.shape
        length = (~seq_mask).sum(-1)
        
        packed_x = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(2, B, self.hidden_size // 2, dtype=torch.float32, device=cond_z.device) # <2, B, hidden_size>, use bidirectional rnn
        w_feats, hn = self.sequential_ecoder(packed_x, h0) # _ , <2, B, hidden_size>
        w_feats, max_ = nn.utils.rnn.pad_packed_sequence(w_feats, batch_first=True, total_length=nframe) # <B, L, 2 * hidden_size>
        w_feats = w_feats.contiguous()

        B, L, H = w_feats.size()
        idx = (length-1).long() # 0-indexed
        idx = idx.view(B, 1, 1).expand(B, 1, H//2)
        fLSTM = w_feats[:,:,:H//2].gather(1, idx).view(B, H//2) # <B, hidden_size // 2>
        bLSTM = w_feats[:,0,H//2:].view(B,H//2) # <B, hidden_size // 2>
        s_feats = torch.cat([fLSTM, bLSTM], dim=1) # <B, hidden_size>
        
        hidden = torch.cat([s_feats, cond_z], dim=1) # <B, D>, D = hidden_size + condition_size
        hidden = self.res_module(hidden) # <B, D>

        mu = self.mu_encoder(hidden) # <B, z_dim>
        logvar = self.logvar_encoder(hidden) # <B, z_dim>

        return mu, logvar

class MotionDecoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, condition_latent_size: int, z_size: int=32):
        super(MotionDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.condition_latent_size = condition_latent_size
        self.z_size = z_size

        ## layer for fusing latent and condition
        self.latent_linear_layer = nn.Linear(self.z_size + self.condition_latent_size, self.hidden_size)
        
        ## transformer decoder for human pose
        self.num_heads = 8
        self.ff_size = 1024
        self.num_layers = 2
        self.dropout = 0.1
        self.activation = "gelu"
        self.sequence_pos_encoder = PositionalEncoding(self.hidden_size, dropout=0.0)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.hidden_size,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

        ## final output layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    
    def forward(self, z_h, cond_z, tgt_mask):
        B, nframe = tgt_mask.shape

        rec_hidden = torch.cat([z_h, cond_z], dim=1) # <B, z_size + condition_size>
        rec_hidden = self.latent_linear_layer(rec_hidden) # <B, hidden_size>
        rec_hidden = rec_hidden.unsqueeze(dim=0) # <1, B, D>, sequence size is 1

        ## positional encoding as query
        timequeries = torch.zeros(nframe, B, self.hidden_size, device=z_h.device)
        timequeries = self.sequence_pos_encoder(timequeries) # <nframe, B, D>
        output = self.seqTransDecoder(tgt=timequeries, memory=rec_hidden, tgt_key_padding_mask=tgt_mask) # <nframe, B, D>

        rec_x = self.output_layer(output) # <nframe, B, h_dim>

        return rec_x

class MotionModel(nn.Module):
    def __init__(self, config):
        super(MotionModel, self).__init__()

        self.config = config
        self.device = self.config.device

        self.encoder = MotionEncoder(self.config.input_size, self.config.hidden_size, self.config.condition_latent_size, self.config.z_size)
        self.decoder = MotionDecoder(self.config.input_size, self.config.hidden_size, self.config.condition_latent_size, self.config.z_size)
    
    def _sampler(self, mu, logvar):
        var = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.to(self.device)
        return eps.mul(var).add_(mu)

    def forward(self, x, cond_z, seq_mask):
        mu, logvar = self.encoder(x, cond_z, seq_mask)
        z_h = self._sampler(mu, logvar)
        rec_x = self.decoder(z_h, cond_z, seq_mask)

        return rec_x, mu, logvar
    
    def sample(self, cond_z, k: int=1, nframe_type: str='fixed', nframe: int=60):
        B = cond_z.shape[0] # must be 1
        assert B == 1, 'Batch size in eval must be 1.'
        sample_res = []
        sample_mask = []

        for i in range(k):
            z_h = torch.randn([B, self.config.z_size], dtype=torch.float32).cuda()

            if nframe_type == 'fixed':
                seq_mask = torch.zeros(B, self.config.max_motion_len, dtype=torch.bool).to(self.device)
                seq_mask[:, nframe:] = True
            elif nframe_type == 'rand':
                seq_mask = torch.zeros(B, self.config.max_motion_len, dtype=torch.bool).to(self.device)
                for j in range(B):
                    rn = random.randint(30, 120)
                    seq_mask[j, rn:] = True
            else:
                raise Exception('Unexcepted nframes type.')

            rec_x = self.decoder(z_h, cond_z, seq_mask)

            ## decode
            sample_res.append(rec_x.unsqueeze(1)) # [<S, 1, B, D>, ...]
            sample_mask.append(seq_mask.unsqueeze(1)) # [<B, 1, S>, ...]
        
        return torch.cat(sample_res, dim=1), torch.cat(sample_mask, dim=1) # <S, K, B, D>, <B, K, S>
