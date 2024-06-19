from senticnet.senticnet import SenticNet
import numpy as np
import torch


# sn = SenticNet()
class TextSentiment(object):
    def __init__(self):
        self.sn = SenticNet()
        
    def call(self, texts, tokenizer, device):
        res = []
        emos = ''
        for text in texts:
            word_list = tokenizer.tokenize(text)
            text_degree_senti_res = []
            text_class_senti_res = []
            text_class_emotion_res = []
            for word in word_list:
                try:
                    word_emotion_class = [m.replace('#','') for m in self.sn.moodtags(word)]
                    word_polarity_value = float(self.sn.concept(word)['polarity_value'])
                    word_polarity_class = self.sn.polarity_label(word)
                except:
                    word_emotion_class = ['Neutral','Neutral']
                    word_polarity_value = float(0)
                    word_polarity_class = 'Neutral'
                if word_polarity_value != 0:
                    emos+=word+' '
                text_degree_senti_res.append(word_polarity_value)
                text_class_senti_res.append(word_polarity_class)
                text_class_emotion_res.append(word_emotion_class)
            res.append([torch.tensor(text_degree_senti_res).to(device), text_class_senti_res, text_class_emotion_res, emos])
        return res
    