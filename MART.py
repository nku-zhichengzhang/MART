from einops import rearrange, reduce, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torchsummary import summary
import math
import random
from skimage.feature import hog
from skimage import io
from skimage import data
import numpy as np
from models.mbt.VanillaViT import DropPath, Mlp, Block, get_sinusoid_encoding_table
from models.text import get_text_model

def AttMask(attention, masking_prob, masking_mode, masking_ratio, show_ratio, show_max):
    
    # Get AttMask (High, Hints or Low)
    masks = get_mask(attention,
                     masking_prob,
                     masking_mode,
                     masking_ratio
                     )
    
    # For AttMask-Hints, randomly reveal some of the most highly attended tokens
    if masking_mode == 'attmask_hint':
        
        # Get a mask of the top show(%) most attended tokens
        top_masks = get_mask(attention,
                             1,
                             masking_mode,
                             show_max
                             )
    
        # Reveal some of the most attended tokens
        masks = show_hints(top_masks, masks, show_ratio)
    
    return masks


def get_mask(attention, masking_prob, masking_mode, masking_ratio):
    
    # Token masking
    token_mask = attention_masking(attention, masking_mode, masking_ratio)

    # Mask a subset based on masking_prob threshold
    generator = torch.rand(attention.shape[0], device=attention.device)
    token_mask[generator > masking_prob] = False

    return token_mask


def attention_masking(attention, masking_mode, masking_ratio):

    N = int(attention.shape[1]*masking_ratio)
    attn_mask = torch.zeros(attention.shape, dtype=torch.bool, device = attention.device)

    if masking_mode in ['attmask_high', 'attmask_hint']:
        idx = torch.argsort(attention, descending=True)[:,:N]
    elif masking_mode == 'attmask_low':
        idx = torch.argsort(attention, descending=False)[:,:N]
    else:
        raise('Use attmask_high, attmask_hint or attmask_low')
    
    attn_mask.scatter_(1, idx, True)
    
    return attn_mask


def show_hints(top_masks, masks, show_ratio):

    _, n_tokens = masks.shape
    reveal_tokens = int(show_ratio*n_tokens)

    selected_high = torch.multinomial(top_masks.float(), reveal_tokens)

    masks.scatter_(1, selected_high, False)

    return masks

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttentionOp(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, num_layer=4, activation='ReLU'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.activation = getattr(nn, activation)
        map=[]
        for _ in range(num_layer-1):
            map.append(nn.Conv1d(in_channels=self.y_size, out_channels=self.y_size, kernel_size=3,
                      stride=1, padding=1))
            map.append(self.activation())
        map.append(nn.Conv1d(in_channels=self.y_size, out_channels=self.x_size, kernel_size=3,
                      stride=1, padding=1))
        map.append(self.activation())
        self.net = nn.Sequential(*map)
        
    # Ours
    def forward(self, x, y):
        """Calulate the score 
        """
        T = x.size(1)
        # tmesh = time_mesh(T, x.device)
        x_pred = torch.flatten(self.net(y.permute(0,2,1)).permute(0,2,1),start_dim=0,end_dim=1).contiguous()    # bs x T, emb_size
        x = torch.flatten(x,start_dim=0,end_dim=1).contiguous() # bs x T, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = (pos - neg).mean()
        return nce

class TemporalAffectiveComplementaryLearning(nn.Module):
	def __init__(self, dim, clsnum, head=8, drop_path=.01):
		super().__init__()
		self.head = head
		self.v2l_attn=MultiHeadAttentionOp(in_features=dim, head_num=head)
		self.a2l_attn=MultiHeadAttentionOp(in_features=dim, head_num=head)
		self.v_norm = nn.LayerNorm(dim)
		self.a_norm = nn.LayerNorm(dim)
		self.l_norm = nn.LayerNorm(dim)
		self.drop_path = DropPath(drop_path)
		self.v_proj = Mlp(dim)
		self.a_proj = Mlp(dim)
		self.v2a_contrast = CPC(768,768)
		self.a2v_contrast = CPC(768,768)
		self.v_fc = Mlp(dim, out_features=clsnum)
		self.a_fc = Mlp(dim, out_features=clsnum)
		

	def forward(self, fv, fa, fl, fl_s, Ns):
		fv = self.v_norm(fv)
		fa = self.a_norm(fa)
		fl = self.l_norm(fl)

		v2l, Vatt = self.v2l_attn(q=fl, k=fv, v=fv)
		a2l, Aatt = self.a2l_attn(q=fl, k=fa, v=fa)

		Vatt = rearrange(Vatt, '(b h) v l ->b h v l', h=self.head).mean(dim=1)
		Aatt = rearrange(Aatt, '(b h) a l ->b h a l', h=self.head).mean(dim=1)
		
		v2l = self.v_proj(self.drop_path(v2l))
		a2l = self.a_proj(self.drop_path(a2l))
		
		# Pull loss
		loss_v = self.v2a_contrast(v2l, a2l)
		loss_a = self.a2v_contrast(a2l, v2l)
		loss_comp = loss_v + loss_a

		# cls logits
		v2l = rearrange(v2l, '(b s) n d ->b s n d', s=Ns)
		a2l = rearrange(a2l, '(b s) n d ->b s n d', s=Ns)
		v_c = self.v_fc(v2l.mean(dim=1).mean(dim=1))
		a_c = self.a_fc(a2l.mean(dim=1).mean(dim=1))

		return loss_comp, Vatt, Aatt, v_c, a_c

class MART(nn.Module):
 
	def __init__(self,
				 img_size=112,
				 num_frames=16,
				 input_channels=3,
				 feature_dim=2*3*16*16,
				 patch_embed_dim=768,
				 conv_patch_embed_kernel=(3, 7, 7),
				 conv_patch_embed_stride=(2, 4, 4),
				 conv_patch_embed_padding=(1, 3, 3),
				 embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
				 pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
				 pool_kv_stride_adaptive=[1, 8, 8],
				 pool_kvq_kernel=[3, 3, 3],
				 head=None,
				 decoder_emb=512,
				 pretrain_pth=None,
				 model=None,
				 **kwargs):

		super().__init__()
		self.num_frames = num_frames
		self.img_size = img_size
		self.vit = model
		self.stride = conv_patch_embed_stride
		self.downsample_rate = 2 ** len(pool_q_stride_size)
		self.embed_dims = 2**len(embed_dim_mul) * patch_embed_dim
		self.lan_model = get_text_model(useLarge=False)
		self.tacl = TemporalAffectiveComplementaryLearning(dim=patch_embed_dim, clsnum=model.n_classes)
		
		in_features = 768 #self.vit.norm_embed.normalized_shape[0]，这里要改为norm层参数
		self.decoder_embd = nn.Linear(in_features, decoder_emb, bias=True)
		self.decoder_pred = nn.Linear(decoder_emb, feature_dim, bias=True)
		self.decoder_blocks = nn.ModuleList([Block(decoder_emb, 16, qkv_bias=True, init_values=0.)  for i in range(8)])
		self.Vmask_token = nn.Parameter(torch.zeros(1, 1, patch_embed_dim))
		self.decoder_pos_embed = get_sinusoid_encoding_table(392, decoder_emb)
		self.decoder_norm = nn.LayerNorm(decoder_emb)

		nn.init.xavier_uniform_(self.decoder_pred.weight)
		nn.init.constant_(self.decoder_pred.bias, 0)
		nn.init.trunc_normal_(self.Vmask_token, std=.02)

		self.decoder_norm.apply(self.init_weights)
		self.decoder_embd.apply(self.init_weights)
		self.decoder_pred.apply(self.init_weights)
		self.decoder_blocks.apply(self.init_weights)
		
	
	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
	
	@torch.jit.ignore
	def no_weight_decay_keywords(self):
		return {'pos_embed', 'cls_token', 'mask_token'}



	def forward_d_att(self, v, Vattm, mratio):
		# Visual
		# get attention
		# generate attention mask
		Vmask = AttMask(Vattm, 0.5, 'attmask_hint', mratio, 0.1 * mratio, 0.1)
		# mask tokens
		B, L, C = v.shape
		Vmask_token = self.Vmask_token.expand(B, L, -1)
		w = Vmask.flatten(1).unsqueeze(-1).type_as(Vmask_token)
		v = v * (1 - w) + Vmask_token * w

		# Decoder
		# position
		v = self.decoder_embd(v)
		v = v + self.decoder_pos_embed.to(v.device)

		# Blocks
		for blk in self.decoder_blocks:
			v = blk(v)
		v = self.decoder_norm(v)
		v = self.decoder_pred(v)

		return v, Vmask

	def forward_d(self, visual):
		x = self.decoder_embd(visual)
		x = x + self.decoder_pos_embed.to(x.device)

		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)
		x = self.decoder_pred(x)
		return x

	def forward_c(self, visual, audio, seq_len, batch, Ts, tau=10):
		# patch pool
		visual = visual.mean(dim=1)
		audio = audio.mean(dim=1)

		# temporal segment pool
		visual = visual.view(seq_len, batch, -1).transpose(0,1).contiguous()
		visual = torch.mean(visual, dim=1) # B, D
		audio = audio.view(Ts//self.vit.audio_time, batch, -1).transpose(0,1).contiguous()
		audio = torch.mean(audio, dim=1) # B, D
        
        # fc classifier
		output_a = self.vit.a_fc(audio)
		output_v = self.vit.v_fc(visual)

		return output_a, output_v

	def forward(self, visual, audio, language, target_recon, target, mratio, cls_loss, isTrain=True, lamb=.9):
		# Feature extraction
		fv, fa, fl_w, fl_s, Ns, Nb, Ta, fv_att, fa_att = self.forward_features(visual, audio, language, isTrain)
		fv_c, fv_p = fv[:,:1], fv[:,1:]
		fa_c, fa_p = fa[:,:2], fa[:,2:]
		del fv, fa
		torch.cuda.empty_cache()

		# Intra attention
		fv_att_intra = fv_att[:,:,:1,1:].mean(dim=1).mean(dim=1).detach().clone()
		del fv_att, fa_att
		torch.cuda.empty_cache()

		# Complementary learning
		loss_comp, lv_att, la_att, lvc, lva = self.tacl(fv_p, fa_p, fl_w, fl_s, Ns)
		loss_cls_c = cls_loss([lvc, lva], target)
		del fl_w, fl_s, lvc, lva
		torch.cuda.empty_cache()

		# Inter attention
		fv_att_inter = lv_att.mean(dim=1).detach().clone()
		Vattm = lamb * fv_att_intra + (1-lamb) * fv_att_inter
		del lv_att, la_att, fv_att_intra, fv_att_inter
		torch.cuda.empty_cache()

		# Classification
		logits = self.forward_c(fv_c, fa_c, Ns, Nb, Ta)
		loss_cls = cls_loss(logits, target)

		# reconstruction
		recv, maskv = self.forward_d_att(fv_p, Vattm, mratio)
		loss_recv = ((recv - target_recon) ** 2).mean(dim=-1)
		loss_recv = (loss_recv * maskv).sum() / (maskv.sum() + 1e-5)

		return loss_recv, loss_comp, loss_cls, loss_cls_c
	
	def forward_features(self, visual, audio, language, isTrain, mask=None):
		# visual pre
		visual = visual.transpose(0, 1).contiguous()
		visual.div_(self.vit.NORM_VALUE).sub_(self.vit.MEAN)
		Ns, batch, nc, snippet_duration, sample_size, _ = visual.size()
		visual = visual.view(Ns * batch, nc, snippet_duration, sample_size, sample_size).contiguous()

		# visual patchify
		visual = self.vit.vid_model.patch_embed(visual)
		
		# audio pre
		_, Ts, _ = audio.size()
		audio = audio.transpose(0, 1).contiguous()
		audio = audio.chunk(Ts//self.vit.audio_time, dim=0)
		audio = torch.stack(audio, dim=0).contiguous()
		audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
		audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]

		# forward network
		fv, fa, fv_att, fa_att = self.vit.forward_bottleneck_w_VMask_wo_patchify_w_Att(visual,audio)

		# forward language feature
		resl = self.lan_model(language, returnembed=True)
		# word-level and segment-level text embedding
		fl_w, fl_s = resl['embeddings'], resl['cls']
		# fl_w = rearrange(fl_w, '(b s) w d -> b s w d', s=Ns)
		# fl_c = rearrange(fl_c, '(b s) d -> b s d', s=Ns)

		return fv, fa, fl_w, fl_s, Ns, batch, Ts, fv_att, fa_att