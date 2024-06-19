from MART import MART

from opts import parse_opts

from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform, ModelEMA, setup_seed
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader, get_val_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from train import train_epoch_parrel_ema
from train import train_epoch_ema
from validation import val_epoch_av

from torch.utils.data import DataLoader
from torch.cuda import device_count
from tools.word_utils import initialize_tokenizer

from tensorboardX import SummaryWriter

import os, torch

import warnings
warnings.filterwarnings('ignore')



def main():
    opt = parse_opts()
    setup_seed()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    print(opt)
    model, parameters = generate_model(opt)
    model_ema = ModelEMA(model,decay=0.999)
    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    # opt.exp_name = os.path.join('results',opt.exp_name)
    if not os.path.exists(opt.exp_name):
        os.makedirs(opt.exp_name)
    writer = SummaryWriter(logdir=opt.exp_name)

    tokenizer, max_input_length, init_token_idx, eos_token_idx, _, _ = initialize_tokenizer()
    text_tools = {
        'tokenizer': tokenizer,
        'max_input_length': max_input_length,
        'init_token_idx': init_token_idx,
        'eos_token_idx': eos_token_idx
    }

    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform,  opt.seq_len)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.val_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform, opt.val_len)
    val_loader = get_val_loader(opt, validation_data, shuffle=False)
    best_acc = 0
    best_epoch = -1
    vid_model = MART(model=model).cuda()
    for i in range(1, opt.n_epochs + 1):
        train_epoch_parrel_ema(i, train_loader, vid_model, model_ema, criterion, optimizer, opt, training_data.class_names, writer, text_tools)
    model = vid_model.vit
    del vid_model
    torch.cuda.empty_cache()
    for i in range(opt.n_epochs + 1, opt.n_epochs + 151):
        train_epoch_ema(i, train_loader, model.cuda(), model_ema, criterion, optimizer, opt, training_data.class_names, writer)
        acc = val_epoch_av(i, val_loader, model_ema.ema, criterion, opt, writer, optimizer)
        if acc>best_acc:
            best_acc = acc
            best_epoch = i
            torch.save(model_ema.ema.state_dict(), os.path.join(opt.exp_name, 'best.ckpt'))
        print('History Best Accuracy: ', best_acc, '   Best Epoch: ', best_epoch)
    writer.close()


if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""