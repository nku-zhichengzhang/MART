from copy import deepcopy
from functools import partial
from turtle import forward
import numpy as np
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from collections import OrderedDict
from ..text import get_text_model
from ..ast.ast_models import ASTModel
from .VanillaViT import vit_base_patch16_112 as ViT



class MBT(nn.Module):
    def __init__(self, n_classes, audio_time=100, r_act=4, senti_class=2):
        super(MBT, self).__init__()
        self.n_classes = n_classes
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE
        self.r_act = r_act
        self.audio_time = audio_time
        self.ast_model = ASTModel(input_tdim=100)
        self.vid_model = ViT()
        self.lan_model = get_text_model(useLarge=False)

        self.a_fc = nn.Linear(768, self.n_classes)
        self.v_fc = nn.Linear(768, self.n_classes)
        self.a_s_fc = nn.Linear(768, senti_class)
        self.v_s_fc = nn.Linear(768, senti_class)
        self.bottleneck = nn.Parameter(torch.zeros(1, 32, 768))

        trunc_normal_(self.bottleneck, std=.02)


    def forward_tfn(self, visual, audio):
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        k = seq_len // self.r_act + 1
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()

        # with torch.no_grad():
        visual_feature = self.vid_model.forward_features(visual)
        # visual_feature = self.vid_model.forward_freeze_features(visual)
        visual_feature = visual_feature.view(seq_len, batch, -1).transpose(0,1).contiguous()
        visual_feature = torch.mean(visual_feature, dim=1) # B, D

        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model(audio, extractEmb=True)
        audio_feature = audio_feature.view(Ts//self.audio_time, batch, -1).transpose(0,1).contiguous()
        audio_feature = torch.mean(audio_feature, dim=1) # B, D
        # Fusion
        output_a = self.a_fc(audio_feature)
        output_v = self.v_fc(visual_feature)
        # output = (output_a + output_v) / 2
        # if istrain:
        return [output_a, output_v]

    
    def bottleneck_fusion(self, visual, audio, bottles, extractFea=False, extractVA=False):
        bv,tv,_ = visual.size()
        _,ta,_ = audio.size()
        _,tb,_ = bottles.size()
        for v_blk,a_blk in zip(self.vid_model.blocks[8:], self.ast_model.v.blocks[8:]):
            
            ab = torch.cat([audio, bottles],dim=1)
            ab = a_blk(ab)
            audio = ab[:,:ta]
            bottlesa = ab[:,ta:]
            
            vb = torch.cat([visual, bottles],dim=1)
            vb = v_blk(vb)
            visual = vb[:,:tv]
            bottlesv = vb[:,tv:]
            bottles = (bottlesa+bottlesv)/2
        # output
        if extractFea:
            return self.vid_model.norm(visual)
        if extractVA:
            return self.vid_model.norm(visual), self.ast_model.v.norm(audio)
        audio = self.ast_model.v.norm(audio)
        audio = (audio[:, 0] + audio[:, 1]) / 2
        
        visual = self.vid_model.norm(visual)
        # if self.vid_model.fc_norm is not None:
        #     visual = self.vid_model.fc_norm(visual.mean(1))
        # else:
        visual = visual[:, 0]
        
        return visual, audio
    
    def bottleneck_fusion_cls(self, visual, audio, bottles, extractFea=False):
        bv,tv,_ = visual.size()
        _,ta,_ = audio.size()
        _,tb,_ = bottles.size()
        for v_blk,a_blk in zip(self.vid_model.blocks[8:], self.ast_model.v.blocks[8:]):
            
            ab = torch.cat([audio, bottles],dim=1)
            ab = a_blk(ab)
            audio = ab[:,:ta]
            bottlesa = ab[:,ta:]
            
            vb = torch.cat([visual, bottles],dim=1)
            vb = v_blk(vb)
            visual = vb[:,:tv]
            bottlesv = vb[:,tv:]
            bottles = (bottlesa+bottlesv)/2
        # output
        if extractFea:
            return self.vid_model.norm(visual)
        audio = self.ast_model.v.norm(audio)
        audio = audio[:, :2].mean(dim=1)
        
        visual = self.vid_model.norm(visual)
        if self.vid_model.fc_norm is not None:
            visual = self.vid_model.fc_norm(visual[:, :1]).mean(dim=1)
        else:
            visual = visual[:, 0]
        
        return visual, audio
    
    def audio_attention(self, attLayer, x):
        B, N, C = x.shape
        qkv = attLayer.qkv(x).reshape(B, N, 3, attLayer.num_heads, C // attLayer.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attLayer.scale
        attn = attn.softmax(dim=-1)
        attn = attLayer.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attLayer.proj(x)
        x = attLayer.proj_drop(x)

        return x, attn

    def audio_block_wAtt(self, block, x):
        res = x
        x = block.norm1(x)
        x, attn = self.audio_attention(block.attn, x)
        x = block.drop_path(x) + res
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        return x, attn

    def bottleneck_fusion_cls_attn(self, visual, audio, bottles, extractFea=False):
        bv,tv,_ = visual.size()
        _,ta,_ = audio.size()
        _,tb,_ = bottles.size()
        for id, (v_blk,a_blk) in enumerate(zip(self.vid_model.blocks[8:], self.ast_model.v.blocks[8:])):
        
            if id < len(self.ast_model.v.blocks[8:]) - 1:
                ab = torch.cat([audio, bottles],dim=1)
                ab = a_blk(ab)
                audio = ab[:,:ta]
                bottlesa = ab[:,ta:]
                
                vb = torch.cat([visual, bottles],dim=1)
                vb = v_blk(vb)
                visual = vb[:,:tv]
                bottlesv = vb[:,tv:]
                bottles = (bottlesa+bottlesv)/2
            else:
                ab = torch.cat([audio, bottles],dim=1)
                ab = a_blk(ab)
                audio = ab[:,:ta]
                bottlesa = ab[:,ta:]
                
                vb = torch.cat([visual, bottles],dim=1)
                vb, attn = v_blk(vb, return_attention=True)
                attn = attn[:,:,:tv,:tv]
                visual = vb[:,:tv]
                bottlesv = vb[:,tv:]
                bottles = (bottlesa+bottlesv)/2
        # output
        if extractFea:
            return self.vid_model.norm(visual), attn
        audio = self.ast_model.v.norm(audio)
        audio = (audio[:, 0] + audio[:, 1]) / 2
        
        visual = self.vid_model.norm(visual)
        if self.vid_model.fc_norm is not None:
            visual = self.vid_model.fc_norm(visual[:, 0])
        else:
            visual = visual[:, 0]
        
        return visual, audio

    def bottleneck_fusion_cls_VA_attn(self, visual, audio, bottles, extractFea=False):
        bv,tv,_ = visual.size()
        _,ta,_ = audio.size()
        _,tb,_ = bottles.size()
        for id, (v_blk,a_blk) in enumerate(zip(self.vid_model.blocks[8:], self.ast_model.v.blocks[8:])):
        
            if id < len(self.ast_model.v.blocks[8:]) - 1:
                ab = torch.cat([audio, bottles],dim=1)
                ab = a_blk(ab)
                audio = ab[:,:ta]
                bottlesa = ab[:,ta:]
                
                vb = torch.cat([visual, bottles],dim=1)
                vb = v_blk(vb)
                visual = vb[:,:tv]
                bottlesv = vb[:,tv:]
                bottles = (bottlesa+bottlesv)/2
            else:
                ab = torch.cat([audio, bottles],dim=1)
                ab, Aattn = self.audio_block_wAtt(a_blk, ab)
                Aattn = Aattn[:,:,:ta,:ta]
                audio = ab[:,:ta]
                bottlesa = ab[:,ta:]
                
                vb = torch.cat([visual, bottles],dim=1)
                vb, Vattn = v_blk(vb, return_attention=True)
                Vattn = Vattn[:,:,:tv,:tv]
                visual = vb[:,:tv]
                bottlesv = vb[:,tv:]
                bottles = (bottlesa+bottlesv)/2
        # output
        visual = self.vid_model.norm(visual)
        audio = self.ast_model.v.norm(audio)
        if extractFea:
            return visual, audio, Vattn, Aattn
        
        audio = (audio[:, 0] + audio[:, 1]) / 2
        
        if self.vid_model.fc_norm is not None:
            visual = self.vid_model.fc_norm(visual[:, 0])
        else:
            visual = visual[:, 0]
        
        return visual, audio

    def forward_bottleneck(self, visual, audio, extractFea=False):
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        k = seq_len // self.r_act + 1
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()

        # with torch.no_grad():
        visual_feature = self.vid_model.forward_features_fore(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature = self.bottleneck_fusion(visual_feature, audio_feature, bottles)

        
        # visual_feature = self.vid_model.forward_freeze_features(visual)
        visual_feature = visual_feature.view(seq_len, batch, -1).transpose(0,1).contiguous()
        visual_feature = torch.mean(visual_feature, dim=1) # B, D

        
        audio_feature = audio_feature.view(Ts//self.audio_time, batch, -1).transpose(0,1).contiguous()
        audio_feature = torch.mean(audio_feature, dim=1) # B, D
        
        # fc classifier
        output_a = self.a_fc(audio_feature)
        output_v = self.v_fc(visual_feature)
        # output = (output_a + output_v) / 2
        # if istrain:
        return [output_a, output_v]
    
    
    
    def forward_bottleneck_cls(self, visual, audio, extractFea=False):
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        k = seq_len // self.r_act + 1
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()

        # with torch.no_grad():
        visual_feature = self.vid_model.forward_features_fore_cls(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature = self.bottleneck_fusion_cls(visual_feature, audio_feature, bottles)
        
        # visual_feature = self.vid_model.forward_freeze_features(visual)
        visual_feature = visual_feature.view(seq_len, batch, -1).transpose(0,1).contiguous()
        
        # no averagepooling in temporal dimension
        temporal_v = visual_feature
        
        visual_feature = torch.mean(visual_feature, dim=1) # B, D

        
        audio_feature = audio_feature.view(Ts//self.audio_time, batch, -1).transpose(0,1).contiguous()
        
        # no averagepooling in temporal dimension
        temporal_a = audio_feature
        
        audio_feature = torch.mean(audio_feature, dim=1) # B, D
        
        # fc classifier
        output_a = self.a_fc(audio_feature)
        output_v = self.v_fc(visual_feature)
        # output = (output_a + output_v) / 2
        # if istrain:
        
        with torch.no_grad():
            temporal_score_v = self.v_fc(temporal_v)  # [bs, seqlen, class]
            temporal_score_a = self.a_fc(temporal_a)  # [bs, seqlen, class]
            temporal_score = temporal_score_v + temporal_score_a
        
        return [output_a, output_v], temporal_score
    
    def forward_bottleneck_w_VMask_wo_patchify(self, visual, audio):
        # visual
        visual_feature = self.vid_model.forward_features_mid_cls(visual)

        # audio
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)

        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature = self.bottleneck_fusion(visual_feature, audio_feature, bottles, extractVA=True)

        return visual_feature, audio_feature

    def forward_bottleneck_w_VMask_wo_patchify_w_Att(self, visual, audio):
        # visual
        visual_feature = self.vid_model.forward_features_mid_cls(visual)

        # audio
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)

        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature, Vattn, Aattn = self.bottleneck_fusion_cls_VA_attn(visual_feature, audio_feature, bottles, extractFea=True)

        return visual_feature, audio_feature, Vattn, Aattn
    
    def forward_bottleneck_w_VMask_wo_patchify_w_Att_w_Lan(self, visual, audio):
        # visual
        visual_feature = self.vid_model.forward_features_mid_cls(visual)

        # audio
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)

        assert audio_feature.shape[0] == visual_feature.shape[0]

        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature, Vattn, Aattn = self.bottleneck_fusion_cls_VA_attn(visual_feature, audio_feature, bottles, extractFea=True)

        # language
        language_feature = self.lan_model(language,returnembed=False)
        

        return visual_feature, audio_feature, Vattn, Aattn

    def forward_bottleneck_w_VMask(self, visual, audio):
        
        visual_feature = self.vid_model.forward_features_mid(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature = self.bottleneck_fusion(visual_feature, audio_feature, bottles, extractFea=True)

        return visual_feature

    def forward_bottleneck_w_VMask_cls(self, visual, audio):
        
        visual_feature = self.vid_model.forward_features_mid_cls(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature = self.bottleneck_fusion(visual_feature, audio_feature, bottles, extractFea=True)
        return visual_feature

    
    def forward_bottleneck_w_VMask_att(self, visual, audio):
        
        visual_feature = self.vid_model.forward_features_mid_cls(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature, audio_feature, Vattn, Aattn = self.bottleneck_fusion_cls_attn(visual_feature, audio_feature, bottles, extractFea=True)
        # if extractFea:
        return visual_feature, audio_feature, Vattn, Aattn
    
    def forward_bottleneck_w_VMask_fea(self, visual, audio):
        
        visual_feature = self.vid_model.forward_features_mid_cls(visual)
        
        # audio
        bs, Ts, Ds = audio.size()
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(Ts//self.audio_time, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        with torch.no_grad():
            audio_feature = self.ast_model.forward_fea(audio, extractEmb=True)
            
        assert audio_feature.shape[0] == visual_feature.shape[0]
        bottles = self.bottleneck.repeat(audio_feature.shape[0],1,1)
        visual_feature = self.bottleneck_fusion_cls(visual_feature, audio_feature, bottles, extractFea=True)

        return visual_feature
    
    def forward(self, visual, audio, type='mbt'):
        if type=='mbt':
            # return self.forward_bottleneck(visual, audio)
            return self.forward_bottleneck_cls(visual, audio)
        else:
            return self.forward_tfn(visual, audio)