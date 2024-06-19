from numpy import extract
import torch
import torch.nn as nn
import torchvision
from models.visual_stream import VisualStream
from models.ast.ast_models import ASTModel

class VAANet(VisualStream):
    def __init__(self,
                 snippet_duration=16,
                 sample_size=112,
                 n_classes=8,
                 seq_len=10,
                 pretrained_resnet101_path='',
                 audio_embed_size=256,
                 audio_n_segments=10,):
        super(VAANet, self).__init__(
            snippet_duration=snippet_duration,
            sample_size=sample_size,
            n_classes=n_classes,
            seq_len=seq_len,
            pretrained_resnet101_path=pretrained_resnet101_path
        )

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        self.ast = ASTModel(input_tdim=100)

        self.a_fc = nn.Sequential(
            nn.Linear(768, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )

        self.aa_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.audio_n_segments, self.audio_n_segments, bias=True),
            'relu': nn.ReLU(),
        })

        self.av_fc = nn.Linear(self.audio_embed_size + self.hp['k'], self.n_classes)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor):
        visual = visual.transpose(0, 1).contiguous()
        visual.div_(self.NORM_VALUE).sub_(self.MEAN)

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.size()
        visual = visual.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size).contiguous()
        with torch.no_grad():
            F = self.resnet(visual)
            F = torch.squeeze(F, dim=2)
            F = torch.flatten(F, start_dim=2)
        F = self.conv0(F)  # [B x 512 x 16]

        Hs = self.sa_net['conv'](F)
        Hs = torch.squeeze(Hs, dim=1)
        Hs = self.sa_net['fc'](Hs)
        As = self.sa_net['softmax'](Hs)
        As = torch.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = torch.mul(F, torch.unsqueeze(As, dim=1).repeat(1, self.hp['k'], 1))

        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net['conv'](G)
        Hc = torch.squeeze(Hc, dim=1)
        Hc = self.cwa_net['fc'](Hc)
        Ac = self.cwa_net['softmax'](Hc)
        Ac = torch.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = torch.mul(fS, torch.unsqueeze(Ac, dim=2).repeat(1, 1, self.hp['m']))
        fSC = torch.mean(fSC, dim=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        Ht = self.ta_net['conv'](fSC)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = torch.mul(fSC, torch.unsqueeze(At, dim=1).repeat(1, self.hp['k'], 1))
        
        # no averagepooling in temporal dimension
        temporal_v = fSCT
        
        fSCT = torch.mean(fSCT, dim=2)  # [bs x 512]

        # Audio branch
        bs = audio.size(0)
        audio = audio.transpose(0, 1).contiguous()
        audio = audio.chunk(self.audio_n_segments, dim=0)
        audio = torch.stack(audio, dim=0).contiguous()
        audio = audio.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        audio = torch.flatten(audio, start_dim=0, end_dim=1)  # [B x 256 x 32]
        # audio = torch.unsqueeze(audio, dim=1)
        with torch.no_grad():
            audio = self.ast(audio, extractEmb=True)
        # audio = self.a_resnet(audio)
        audio = torch.flatten(audio, start_dim=1).contiguous()
        audio = self.a_fc(audio)
        audio = audio.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        audio = audio.permute(1, 2, 0).contiguous()

        Ha = self.aa_net['conv'](audio)
        Ha = torch.squeeze(Ha, dim=1)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = torch.mul(audio, torch.unsqueeze(Aa, dim=1).repeat(1, self.audio_embed_size, 1))
        
        # no averagepooling in temporal dimension
        temporal_a = fA
        
        fA = torch.mean(fA, dim=2)  # [bs x 256]

        # Fusion
        fSCTA = torch.cat([fSCT, fA], dim=1)
        output = self.av_fc(fSCTA)
        
        temporal_va = torch.cat([temporal_v, temporal_a], dim=1) # [bs, 768, seqlen]
        temporal_va = temporal_va.permute(0, 2, 1)
        with torch.no_grad():
            temporal_score = self.av_fc(temporal_va)  # [bs, seqlen, class]

        return output, alpha, beta, gamma, temporal_score