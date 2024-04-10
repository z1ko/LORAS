from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import lightning
import os
import pandas
import numpy
import lmdb
import pickle

from types import SimpleNamespace

import torch.utils
import torch.utils.data

def load_frame_features(databases, view, start_frame, end_frame, id):
    """ Load a subset of frame embeddings
    """
    elements = []
    with databases[view].begin(write=False) as e:
        for i in range(start_frame, end_frame):
            key = os.path.join(id, f'{view}/{view}_{i:010d}.jpg')
            frame_data = e.get(key.strip().encode('utf-8'))
            if frame_data is None:
                print(f"[!] No data found for key={key}.")
                exit(2)

            frame_data = numpy.frombuffer(frame_data, 'float32')
            elements.append(frame_data)

    features = numpy.array(elements) # [T, D]
    return torch.tensor(features)


def load_frame_poses(path_to_poses, start_frame, end_frame):
    """ Load a subset of frame skeletal poses
    """
    with open(path_to_poses, "rb") as f:
        sample = pickle.load(f)

    # Changes frequency from 60fps to 30fps
    poses = sample['data'][:, start_frame*2:end_frame*2:2, :, :]
    assert(poses.shape[1] != 0)
    return torch.tensor(poses)


def load_segmentation(path_to_segm, actions, target):
    """ Load a sample segmentation
    """
    labels = []
    start_indices = []
    end_indices = []

    with open(path_to_segm, 'r') as f:
        lines = list(map(lambda s: s.split("\n"), f.readlines()))
        for line in lines:
            start, end, lbl = line[0].split("\t")[:-1]
            start_indices.append(int(start))
            end_indices.append(int(end))

            action_id = actions.loc[actions['action_cls'] == lbl, f'{target}_id']
            segm_len = int(end) - int(start)
            labels.append(numpy.full(segm_len, fill_value=action_id.item()))

    segmentation = numpy.concatenate(labels)
    num_frames = segmentation.shape[0]
    
    # start and end frame idx @30fps
    start_frame = min(start_indices)
    end_frame = max(end_indices)
    assert num_frames == (end_frame-start_frame), \
        "Length of Segmentation doesn't match with clip length."
 
    return segmentation, start_frame, end_frame


# Should be able to produce frames embeddings+poses and coarse labels 
class Assembly101Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, config):
        super().__init__()
        path_to_data = 'data/Assembly101'

        # Load frames database
        self.views = {
            view: lmdb.open(f'{path_to_data}/TSM/{view}', readonly=True, lock=False) 
            for view in config.views
        }
        
        self.samples = self.make_dataset(path_to_data, mode, config)

    def make_dataset(self, path_to_data, mode, config):
        annotations_path = os.path.join(path_to_data, 'coarse-annotations')
        poses_path = os.path.join(path_to_data, 'poses', mode)

        # Load samples split
        split_path = os.path.join(path_to_data, f'{mode}.csv')
        split = pandas.read_csv(split_path)

        # Load actions dictionary
        actions_path = os.path.join(path_to_data, 'coarse-annotations', 'actions.csv')
        actions = pandas.read_csv(actions_path)


        dataset = []
        max_len, min_len = 0, 1e9
        for _, entry in split.iterrows():
            sample = entry.to_dict()

            if sample['view'] not in config.views:
                #print('INFO: skipped sample: not in selected views')
                continue

            # Skip strange samples
            if sample['video_id'] in ['nusar-2021_action_both_9026-b04b_9026_user_id_2021-02-03_163855.pkl']:
                continue

            segm_filename = f"{sample['action_type']}_{sample['video_id']}.txt"
            segm_path = os.path.join(annotations_path, "coarse_labels", segm_filename)
            segm, start_frame, end_frame = load_segmentation(segm_path, actions, config.target_label)
            delta = 0

            # Where to find the poses data
            path_to_pose = os.path.join(poses_path, f"{sample['video_id']}.pkl")
            if os.path.exists(path_to_pose):

                sample['path_to_pose'] = path_to_pose
                with open(path_to_pose, "rb") as f:
                    pose = pickle.load(f)
                    sample['pose'] = pose['data']

            else:
                print('INFO: skipped sample: cant find poses')
                continue

            max_len = max(max_len, len(segm))
            min_len = min(min_len, len(segm))

            # Only use clip size if in training
            if mode == 'train' and config.clip_size is not None:
                for beg in range(0, len(segm) - config.clip_size, config.clip_size):
                    end = beg + config.clip_size

                    sample['segm'] = torch.tensor(segm[beg:end]).long()
                    sample['start_frame'] = start_frame + beg
                    sample['end_frame'] = start_frame + end
                    dataset.append(sample)

            else:
                sample['segm'] = torch.tensor(segm).long()
                sample['start_frame'] = start_frame
                sample['end_frame'] = end_frame
                dataset.append(sample)

        print(f'dataset length: {len(dataset)}, max_frames: {max_len}, min_frames: {min_len}')
        return dataset

    def __getitem__(self, idx):
        sample = self.samples[idx]

        view = sample['view']
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        video_id = sample['video_id']
        path_to_pose = sample['path_to_pose']

        features = load_frame_features(self.views, view, start_frame, end_frame, video_id)
        poses =  torch.zeros((1,1)) #load_frame_poses(path_to_pose, start_frame, end_frame)
        return features, poses, sample['segm']

    def __len__(self):
        return len(self.samples)

class Assembly101(lightning.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.validation = Assembly101Dataset('val', config)
        self.training = Assembly101Dataset('train', config)
        self.config = config

    @staticmethod
    def collate(data):
        features, poses, segmentations = [], [], []
        for f, p, s in data:
            features.append(f)
            poses.append(p)
            segmentations.append(s)
        
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        poses = torch.nn.utils.rnn.pad_sequence(poses, batch_first=True)
        segmentations = torch.nn.utils.rnn.pad_sequence(segmentations, batch_first=True, padding_value=-100)
        return features, poses, segmentations

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training,
            self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation,
            self.config.test_batch_size,
            collate_fn=Assembly101.collate,
            pin_memory=True,
        )


if __name__ == '__main__':

    dataset = Assembly101Dataset('train', SimpleNamespace(
        views=['C10095_rgb'],
        clip_size=128,
        target_label='action'
    ))
    
    sample = dataset[0]
    a = 0