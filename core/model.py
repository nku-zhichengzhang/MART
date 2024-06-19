import imp
from logging import raiseExceptions
import torch.nn as nn
from models.vaanet_ast import VAANet
from models.tfn import TFN
from models.MASF import get_default_av_model, MSAFNet
from models.mbt.MBT import MBT
from models.mbt.MBT_w_language import MBT_w_language
def generate_model(opt):
    
    if opt.alg == 'MSAF':
        model_param = get_default_av_model()
        model = MSAFNet(model_param)
        model = model.cuda()
    
    elif opt.alg == 'VAANet':
        model = VAANet(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        )
        model = model.cuda()
        
    elif opt.alg == 'MBT':
        model = MBT(
        n_classes=opt.n_classes,
        audio_time=opt.audio_time,
        r_act=opt.r_act
        )
        model = model.cuda()
        
    elif opt.alg == 'MBT_w_language':
        model = MBT_w_language(
        n_classes=opt.n_classes,
        audio_time=opt.audio_time,
        r_act=opt.r_act
        )
        model = model.cuda()
    
    elif opt.alg == 'TFN':
        model = TFN(
        snippet_duration=opt.snippet_duration,
        sample_size=opt.sample_size,
        n_classes=opt.n_classes,
        seq_len=opt.seq_len,
        audio_embed_size=opt.audio_embed_size,
        audio_n_segments=opt.audio_n_segments,
        pretrained_resnet101_path=opt.resnet101_pretrained,
        )
    else:
        raiseExceptions('Unsupported architecture')

    return model, model.parameters()
