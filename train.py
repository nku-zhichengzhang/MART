from core.utils import AverageMeter, process_data_item_w_language,process_data_item, run_model, calculate_accuracy
from einops import rearrange
from tqdm import tqdm
import time
import numpy as np
from skimage.feature import hog
import math
import random
import torch
from dataset_info import label_info



def train_epoch_parrel_ema(epoch, data_loader, model, model_ema, criterion, optimizer, opt, class_names, writer, text_tools):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_recvs = AverageMeter()
    loss_coms = AverageMeter()
    loss_clss = AverageMeter()
    loss_clscs = AverageMeter()
    accuracies = AverageMeter()
    emotion_accuracies = AverageMeter()
    emotion_origin_accuracies = AverageMeter()
    emotion_F1 = AverageMeter()
    binary_accuracies = AverageMeter()
    binary_origin_accuracies = AverageMeter()
    binary_F1 = AverageMeter()
    class_acc = None


    end_time = time.time()
    mask_ratio_epoch = 0.1 + min(0.5, epoch/opt.n_epochs) * 0.8

    for i, data_item in tqdm(enumerate(data_loader)):
        visual, target, audio, visualization_item, words, batch_size = process_data_item_w_language(opt, data_item)
        data_time.update(time.time() - end_time)

        flattened_words = []
        for vid in range(visual.size(0)):
            for seg in range(visual.size(1)):
                flattened_words.append(words[vid][seg])
                # emotion filter
                emo = text_tools['emo_net'].call([words[vid][seg]],text_tools['tokenizer'],visual.device)[0]
                flattened_words.append(emo[-1])

        text_ids = text_tools['tokenizer'](flattened_words, padding='longest', truncation=True, return_tensors='pt')['input_ids'].to(visual.device)

        B, Ns, Nc, T, H, W = visual.size()
        video=visual.cpu().numpy()
        with torch.no_grad():
            vlabel = torch.from_numpy(video)
            vlabel = rearrange(vlabel, 'b s c (t p0) (h p1) (w p2) -> (b s) (t h w) (p0 p1 p2) c', p0=2, p1=16, p2=16)
            vlabel = (vlabel - vlabel.mean(dim=-2, keepdim=True)
                    ) / (vlabel.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            vlabel = rearrange(vlabel, 'b n p c -> b n (p c)')

        vlabel = vlabel.cuda()
        loss_recv, loss_comp, loss_cls, loss_cls_c = model(visual, audio, text_ids, vlabel, target, mask_ratio_epoch,  criterion)
        loss = loss_recv * 0.05 + loss_comp + loss_cls + loss_cls_c
        loss = loss / opt.accu_step

        losses.update(loss.item(), batch_size)
        loss_recvs.update(loss_recv.item(), batch_size)
        loss_coms.update(loss_comp.item(), batch_size)
        loss_clss.update(loss_cls.item(), batch_size)
        loss_clscs.update(loss_cls_c.item(), batch_size)

        del loss_recv, loss_comp, loss_cls, loss_cls_c
        torch.cuda.empty_cache()

        # Backward and optimize
        loss.backward()
        if ((i + 1) % opt.accu_step == 0) or (i + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('train/batch/loss', losses.val, iter)
        writer.add_scalar('train/batch/loss_recv', loss_recvs.val, iter)
        writer.add_scalar('train/batch/loss_com', loss_coms.val, iter)
        writer.add_scalar('train/batch/loss_cls', loss_clss.val, iter)
        writer.add_scalar('train/batch/loss_cls_c', loss_clscs.val, iter)
        writer.add_scalar('train/batch/acc', accuracies.val, iter)

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))

    writer.add_scalar('train/epoch/loss', losses.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies.avg, epoch)
    
def train_epoch_ema(epoch, data_loader, model, model_ema, criterion, optimizer, opt, class_names, writer):
    print("# ---------------------------------------------------------------------- #")
    print('Training at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    emotion_accuracies = AverageMeter()
    emotion_origin_accuracies = AverageMeter()
    emotion_F1 = AverageMeter()
    binary_accuracies = AverageMeter()
    binary_origin_accuracies = AverageMeter()
    binary_F1 = AverageMeter()
    class_acc = None

    end_time = time.time()

    for i, data_item in tqdm(enumerate(data_loader)):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)

        output, loss, temporal_score = run_model(opt, [visual, target, audio], model, criterion, i, print_attention=False)
        loss = loss / opt.accu_step
        # metrics = calculate_metrics(output, target, opt.dataset)

        accuracies.update(calculate_accuracy(output[0]+output[1], target), batch_size)
        losses.update(loss.item(), batch_size)

        # Backward and optimize
        loss.backward()
        if ((i + 1) % opt.accu_step == 0) or (i + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
            model_ema.update(model)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('train/batch/loss', losses.val, iter)
        writer.add_scalar('train/batch/acc', accuracies.val, iter)

        if opt.debug:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=accuracies))

    # ---------------------------------------------------------------------- #
    print("Epoch Time: {:.2f}min".format(batch_time.avg * len(data_loader) / 60))
    print("Train loss: {:.4f}".format(losses.avg))
    print("Train acc: {:.4f}".format(accuracies.avg))

    writer.add_scalar('train/epoch/loss', losses.avg, epoch)
    writer.add_scalar('train/epoch/acc', accuracies.avg, epoch)