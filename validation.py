from core.utils import AverageMeter, process_data_item, run_model, calculate_accuracy
from dataset_info import label_info
import os
import time
import torch
from sklearn import metrics
import pandas as pd

def val_epoch_av(epoch, data_loader, model, criterion, opt, writer, optimizer):
    print("# ---------------------------------------------------------------------- #")
    print('Validation at epoch {}'.format(epoch))
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()
    
    y_true = []
    y_pred = []
    
    y_sentiment_true = []
    y_sentiment_pred = []
    
    polarity = torch.tensor(label_info[opt.dataset]['sentiment'])

    # df_pred = pd.DataFrame()
    df_pred = pd.read_csv('pred_result.csv')
    
    for i, data_item in enumerate(data_loader):
        visual, target, audio, visualization_item, batch_size = process_data_item(opt, data_item)
        data_time.update(time.time() - end_time)
        with torch.no_grad():
            output, loss, temporal_score, visual_feature = run_model(opt, [visual, target, audio], model, criterion, i, isTrain=False)
            output = output[0]+output[1]

        output_sentiment = polarity[torch.max(output, dim=-1)[1].cpu().detach().tolist()]
        y_sentiment_pred.extend(output_sentiment.cpu().detach().tolist())
        y_sentiment_true.extend(polarity[target].cpu().detach().tolist())
        
        y_true.extend(target.cpu().detach().tolist())
        y_pred.extend(torch.max(output, dim=-1)[1].cpu().detach().tolist())

        pred_dict = {}
        pred_dict['sample_name'] = visualization_item[0][0]
        pred_dict['pred_result'] = y_pred[0]
        pred_dict['ground_truth'] = y_true[0]
        pred_dict['visual_feature'] = visual_feature[0].cpu().numpy()
        pred_dict['model'] = 'MBT'
        df_pred = df_pred.append(pred_dict, ignore_index=True)

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
    
    # df_pred.to_csv('pred_result.csv')
    
    emotion_accuracy = metrics.accuracy_score(y_true, y_pred)
    emotion_F1 = metrics.f1_score(y_true, y_pred, average='weighted')
    emotion_class_discription = metrics.classification_report(y_true, y_pred, output_dict=True)
    
    sentiment_accuracy = metrics.accuracy_score(y_sentiment_true, y_sentiment_pred)
    sentiment_F1 = metrics.f1_score(y_sentiment_true, y_sentiment_pred, average='weighted')
    sentiment_class_discription = metrics.classification_report(y_sentiment_true, y_sentiment_pred, output_dict=True)

    writer.add_scalar('val/loss', losses.avg, epoch)
    
    print("Val loss: {:.4f}".format(losses.avg))
    print('Val emotion acc: {:.4f}'.format(emotion_accuracy))
    print('Val emotion F1: {:.4f}'.format(emotion_F1))
    print('Val emotion class description: {}'.format(emotion_class_discription))
    
    print('Val sentiment acc: {:.4f}'.format(sentiment_accuracy))
    print('Val sentiment F1: {:.4f}'.format(sentiment_F1))
    print('Val sentiment class description: {}'.format(sentiment_class_discription))
    return emotion_accuracy