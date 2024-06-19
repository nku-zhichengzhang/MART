from datasets.ve8 import VE8Dataset
from torch.utils.data import DataLoader


def get_ve8(opt, subset, transforms,audio_n_segments=None):
    spatial_transform, temporal_transform, target_transform = transforms
    return VE8Dataset(opt,
                      opt.video_path,
                      opt.audio_path,
                      opt.annotation_path,
                      opt.srt_path,
                      subset,
                      opt.fps,
                      spatial_transform,
                      temporal_transform,
                      target_transform,
                      need_audio=True,
                      alg=opt.alg,
                      audio_n_segments=audio_n_segments)


def get_training_set(opt, spatial_transform, temporal_transform, target_transform, audio_n_segments=None):

    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'training', transforms, audio_n_segments=None)


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform, audio_n_segments=None):

    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'validation', transforms, audio_n_segments=audio_n_segments)


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):

    transforms = [spatial_transform, temporal_transform, target_transform]
    return get_ve8(opt, 'validation', transforms)


def get_data_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = opt.batch_size if batch_size == 0 else batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl
    )
    
def get_val_loader(opt, dataset, shuffle, batch_size=0):
    batch_size = 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.n_threads,
        pin_memory=True,
        drop_last=opt.dl
    )
