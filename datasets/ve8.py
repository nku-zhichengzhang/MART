import torch
import torch.utils.data as data
import torchaudio
from torchvision import get_image_backend
from transformers import BertTokenizerFast, BertModel
from senticnet.senticnet import SenticNet
from PIL import Image
import random
import json
import os
import functools
import librosa
import numpy as np
import pysrt
from transformers import BertTokenizer
from torchtext.legacy import data
from torchtext.legacy.data import Dataset, Example
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.optim as optim


MAXWORD = 40



def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        return float(input_file.read().rstrip('\n\r'))


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        if value['subset'] == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])
    return video_names, annotations


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    #     with Image.open(f) as img:
    #         return img.convert('RGB')
    return Image.open(path).convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(image_path), "image does not exists"
        video.append(image_loader(image_path))
    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def preprocess_audio(audio_path):
    "Extract audio features from an audio file"
    y, sr = librosa.load(audio_path, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    return mfccs

# for srt
def time2ms(date_time):
    hours = date_time.hour
    minute = date_time.minute
    second = date_time.second
    micros = date_time.microsecond
    ms = micros / 1000 + second * 1000 + minute * 60 * 1000 + hours * 60 * 60 * 1000
    return int(ms)


def read_srt(srt_path):
    srt = pysrt.open(srt_path)    
    
    content = []
    for item in srt.data:
        start = time2ms(item.start.to_time())
        end = time2ms(item.end.to_time()) 
        duration = time2ms(item.duration.to_time()) 
        temp = {
            'count': item.index,
            'start': time2ms(item.start.to_time()),
            'end': time2ms(item.end.to_time()),
            'duration': time2ms(item.duration.to_time()),
            'text': item.text
        }
        if temp['text'] != 'Conversion failed':
            content.append(temp)
    return content
    

def srt2seg(segnum, totaltime, srt_content):
    result = ['' for _ in range(segnum)]
    seglen = totaltime / segnum
    for content in srt_content:
        start = content['start']
        end = content['end']
        duration = content['duration']
        for i in range(segnum):
            if i * seglen <= start and (i + 1) * seglen > start:
                start_idx = i
            if i * seglen < end and (i + 1) * seglen >= end:
                end_idx = i    
        for j in range(start_idx, end_idx + 1):
            result[j] = result[j] + ' ' + content['text']
    return result


class VE8Dataset(data.Dataset):
    def __init__(self,
                 opt,
                 video_path,
                 audio_path,
                 annotation_path,
                 srt_path,
                 subset,
                 fps=30,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 get_loader=get_default_video_loader,
                 need_audio=True,
                 alg='MSAF',
                 audio_n_segments=None
                 ):
        self.subset = subset
        self.data, self.class_names = make_dataset(
            video_root_path=video_path,
            annotation_path=annotation_path,
            audio_root_path=audio_path,
            srt_root_path=srt_path, 
            subset=subset,
            fps=fps,
            need_audio=need_audio
        )
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.fps = fps
        self.ORIGINAL_FPS = 30
        self.need_audio = need_audio
        self.alg = alg
        self.norm_mean = -6.6268077
        self.norm_std = 5.358466
        self.audio_n_segments = opt.audio_n_segments if audio_n_segments is None else audio_n_segments
        # self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.sentinet = SenticNet()

    def __getitem__(self, index):
        data_item = self.data[index]
        video_path = data_item['video']
        frame_indices = data_item['frame_indices']
        snippets_frame_idx = self.temporal_transform(frame_indices)

        if self.need_audio:
            if self.alg == 'VAANet' or self.alg == 'MBT' or self.alg == 'MBT_w_language':
                timeseries_length = 100*self.audio_n_segments
                # audio_path = data_item['audio']
                # feature = preprocess_audio(audio_path).T
                # k = timeseries_length // feature.shape[0] + 1
                # feature = np.tile(feature, reps=(k, 1))
                # audios = feature[:timeseries_length, :]
                # audios = torch.FloatTensor(audios)
                waveform, sr = torchaudio.load(data_item['audio'])
                waveform = waveform - waveform.mean()
                fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
                if fbank.shape[0]<=timeseries_length:
                    k = timeseries_length // fbank.shape[0] + 1
                    fbank = np.tile(fbank, reps=(k, 1))
                    audios = fbank[:timeseries_length, :]
                else:
                    blk = int(fbank.shape[0]/self.audio_n_segments)
                    aud = []
                    for i in list(range(0,fbank.shape[0],blk))[:self.audio_n_segments]:
                        ind = i+int(random.random()*(blk-100))
                        aud.append(fbank[ind:ind+100])
                    audios = torch.cat(aud)
                if audios.shape[0]!=timeseries_length:
                    print(audios.shape)
                audios = torch.FloatTensor(audios)
                if self.subset == 'training':
                    freqm = torchaudio.transforms.FrequencyMasking(24)
                    timem = torchaudio.transforms.TimeMasking(192)
                    audios = torch.transpose(audios, 0, 1)
                    audios = audios.unsqueeze(0)
                    audios = freqm(audios)
                    audios = timem(audios)
                    audios = audios.squeeze(0)
                    audios = torch.transpose(audios, 0, 1)
                audios = (audios - self.norm_mean) / (self.norm_std * 2)
                # if self.subset == 'training':
                #     audios = audios + torch.rand(audios.shape[0], audios.shape[1]) * np.random.rand() / 10
                    # audios = torch.roll(audios, np.random.randint(-10, 10), 0)
                    
            elif self.alg == 'TFN':
                timeseries_length = 4096
                audio_path = data_item['audio']
                feature = preprocess_audio(audio_path).T
                k = timeseries_length // feature.shape[0] + 1
                feature = np.tile(feature, reps=(k, 1))
                audios = feature[:timeseries_length, :]
                audios = torch.FloatTensor(audios)
            elif self.alg == 'MSAF':
                timeseries_length = 212
                audio_path = data_item['audio']
                X, sample_rate = librosa.load(audio_path, duration=2.45, sr=22050 * 2, offset=0.5)
                sample_rate = np.array(sample_rate)
                audios = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
                k = timeseries_length // audios.shape[1] + 1
                audios = np.tile(audios, reps=(1, k))
                audios = audios[:, :timeseries_length]
                audios = torch.FloatTensor(audios)
        else:
            audios = []

        snippets = []
        for snippet_frame_idx in snippets_frame_idx:
            snippet = self.loader(video_path, snippet_frame_idx)
            snippets.append(snippet)

        self.spatial_transform.randomize_parameters()
        snippets_transformed = []
        for snippet in snippets:
            snippet = [self.spatial_transform(img) for img in snippet]
            snippet = torch.stack(snippet, 0).permute(1, 0, 2, 3)
            snippets_transformed.append(snippet)
        snippets = snippets_transformed
        snippets = torch.stack(snippets, 0)
        
        if self.alg == 'MSAF':
            seq_len, c, duration, h, w = snippets.size()
            snippets = snippets.permute(1, 0, 2, 3, 4).contiguous()
            snippets = snippets.view(c, seq_len*duration, h, w).contiguous()

        target = self.target_transform(data_item)
        visualization_item = [data_item['video_id']]
        
        # for srt
        srt_content = read_srt(data_item['srt'])
        waveform, sr = torchaudio.load(data_item['audio'])
        srt_seg = srt2seg(self.audio_n_segments, waveform.shape[1]/sr * 1000, srt_content)
        

        # for srt in srt_seg:
        #     # build the embedding dict for each word in the sentence
        #     words = self.tokenizer.tokenize(srt)
        #     words_embedding = [None for _ in range(MAXWORD)]
        #     # extract corresponding Bert embeddings
        #     # input = self.tokenizer(srt, return_tensors="pt")
        #     # outputs = self.bert(**input).last_hidden_state[:,1:-1]
        #     # num_words.append(len(words))
        #     for wordid, word in enumerate(words[:MAXWORD]):
        #         cnt+=1
        #         # extract sentiment embeddings
        #         try:
        #             concept_info = self.sentinet.concept(word)
        #             polarity_value = float(self.sentinet.polarity_value(word))
        #             polarity_label = self.sentinet.polarity_label(word)
        #             moodtags = [m.replace('#','') for m in self.sentinet.moodtags(word)]
        #             semantics = self.sentinet.semantics(word)
        #             sentics = self.sentinet.sentics(word)
        #             affective_embeddings = [concept_info, polarity_value, polarity_label, moodtags, semantics, sentics]
        #             acnt+=1
        #         except:
        #             affective_embeddings = [None,None,None,None,None,None]

        #         # store the embeddings
        #         words_embedding[wordid]={'word': word, 'affective':affective_embeddings}
        #     for id in range(len(words),MAXWORD):
        #         words_embedding[id]= {'word': 'None', 'affective':[None,None,None,None,None,None]}                
        #     video_words_embedding.append({'embed':words_embedding,'num':len(words)})


        return snippets, target, audios, visualization_item, srt_seg

        


    def __len__(self):
        return len(self.data)


def make_dataset(video_root_path, annotation_path, audio_root_path, srt_root_path, subset, fps=30, need_audio=True):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset) # xx/xx, 'label':'xx'
    class_to_idx = get_class_labels(data) # class_to_idx['label'] = idx
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name # idx_to_class['idx'] = label

    dataset = []
    for i in range(len(video_names)):
        if i % 100 == 0:
            print("Dataset loading [{}/{}]".format(i, len(video_names)))
        video_path = os.path.join(video_root_path, video_names[i])
        if need_audio:
            audio_path = os.path.join(audio_root_path, video_names[i] + '.mp3')
        else:
            audio_path = None
        srt_path = os.path.join(srt_root_path, video_names[i] + '.srt')

        assert os.path.exists(audio_path), audio_path
        assert os.path.exists(video_path), video_path
        assert os.path.exists(srt_path), srt_path

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            print(video_path)
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1],
            'srt': srt_path
        }
        if need_audio: sample['audio'] = audio_path
        assert len(annotations) != 0
        sample['label'] = class_to_idx[annotations[i]['label']]

        ORIGINAL_FPS = 30
        step = ORIGINAL_FPS // fps

        sample['frame_indices'] = list(range(1, n_frames + 1, step))
        dataset.append(sample)
    return dataset, idx_to_class
