import torch
import lightning
import os
import pandas
import numpy
import lmdb
import json
import pickle
import argparse
import einops

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from types import SimpleNamespace
from operator import itemgetter
from tqdm import tqdm

import torch.utils
import torch.utils.data

class Assembly101Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, config):

        # load classes weights
        with open(config.categories_class_weight, 'rb') as f:
            self.weights = [torch.tensor(x, dtype=torch.float32).cuda() 
                            for x in pickle.load(f)]

        self.items = []
        split_path = os.path.join('data/Assembly101/processed', mode)
        for item in tqdm(os.listdir(split_path)):
            item_path = os.path.join(split_path, item)
            
            # No clips
            if config.clip_size == 'None':
                self.items.append((item_path, None))
            else:
                frames = int(item.split('-')[0])
                for beg in range(0, frames - config.clip_size, config.clip_size):
                    self.items.append((item_path, (beg, beg + config.clip_size)))

    def __getitem__(self, idx):
        path, clip = self.items[idx]
        with open(path, 'rb') as f:
            sample = pickle.load(f)

        labels = torch.tensor(sample['fine-labels']).long()
        embeddings = torch.tensor(sample['embeddings'], dtype=torch.float32)
        poses = torch.tensor(sample['poses'],dtype=torch.float32)
        poses = einops.rearrange(poses, 'T H J F -> T (H J) F')

        if clip is not None:
            beg, end = clip

            labels = labels[beg:end, ...]
            embeddings = embeddings[beg:end, ...]
            poses = poses[beg:end, ...]

        return embeddings, poses, labels

    def __len__(self):
        return len(self.items)

class Assembly101(lightning.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.validation = Assembly101Dataset('validation', config)
        self.training = Assembly101Dataset('train', config)
        self.config = config

    @staticmethod
    def collate(data):
        embeddings, poses, labels = [], [], []
        for e, p, l in data:
            embeddings.append(e)
            poses.append(p)
            labels.append(l)
        
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        poses = torch.nn.utils.rnn.pad_sequence(poses, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return embeddings, poses, labels[..., 1:] # take only verb + noun

    @property
    def weights(self):
        return self.training.weights

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.training,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            collate_fn=Assembly101.collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.validation,
            self.config.test_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=Assembly101.collate,
            pin_memory=True,
        )

# =============================================================================================
# PREPROCESS

def gather_split_annotations(annotations, all_frames_dict, jsons_list_poses):
    counters = 0 # Counting total number of segments in all_frames_dict
    for _, aData in tqdm(annotations.iterrows(), 'Populating Dataset', total=len(annotations)):
        # For each segment, find the video id first
        video_id_json = aData.video.strip() + '.json'
        # If the hand poses for the video is not available, skip it
        if not video_id_json in jsons_list_poses:
            continue
        
        # Store segment information as a dictionary
        curr_data = dict()
        curr_data['start_frame'] = aData.start_frame
        curr_data['end_frame'] = aData.end_frame
        curr_data['action'] = aData.action_id
        curr_data['noun'] = aData.noun_id
        curr_data['verb'] = aData.verb_id
        curr_data['action_cls'] = aData.action_cls
        curr_data['toy_id'] = aData.toy_id
        curr_data['shared'] = aData.is_shared

        # Add the dictionary to the list of segments for the video
        all_frames_dict[video_id_json].append(curr_data)
        counters += 1

    print("Inside gather_split_annotations(): ", counters)

def extract_frames(dbs, video_id, frames_count, embeddings):
    # example: 'nusar-2021_action_both_9044-a08_9044_user_id_2021-02-19_083738/C10095_rgb/C10095_rgb_0000011685.jpg'

    # Find correct database
    selected_db = None
    for view, db in dbs:
        with db.begin(write=False) as e:
            key = os.path.join(video_id, f'{view}/{view}_{1:010d}.jpg')
            if e.get(key.strip().encode('utf-8')) is not None:
                selected_db = db
                break

    # No database data found for this video_id
    if selected_db is None:
        return None, None

    complete = True
    database_frames_count = 0
    with selected_db.begin(write=False) as e:
        for i in range(0, frames_count):
            key = os.path.join(video_id, f'{args.view}/{args.view}_{(i+1):010d}.jpg')
            frame_data = e.get(key.strip().encode('utf-8'))
            if frame_data is None:
                print(f"[!] No data found for frame {i}.")
                complete = False
                break

            embeddings[i, :] = numpy.frombuffer(frame_data, 'float32')
            database_frames_count += 1

    # Cap to shortest sequence if possible
    result_range = (0, frames_count)
    if not complete:
        # If only the ending frames are missing, continue
        if frames_count - database_frames_count < 100:
            result_range = (0, database_frames_count)
        else:
            print(f'[!] too many missing frames for {video_id}, skipping sample')
            return None, None

    return embeddings, result_range

def preprocess(args):
    columns = [
        "id", "video", "start_frame", "end_frame", "action_id", "verb_id", "noun_id",
        "action_cls", "verb_cls", "noun_cls", "toy_id", "toy_name", "is_shared"
    ]

    jsons_list_poses = os.listdir(args.path_to_poses)
    jsons_list_poses.sort()

    # NOTE: e2 view requires two possible databases
    if args.egocentric:
        dbs = [
            ('HMC_84355350_mono10bit', lmdb.open(os.path.join(args.path_to_tsm, 'HMC_84355350_mono10bit'))),
            ('HMC_21110305_mono10bit', lmdb.open(os.path.join(args.path_to_tsm, 'HMC_21110305_mono10bit'))),
        ]
    else:
        # Load database for views
        tsm_path = os.path.join(args.path_to_tsm, args.view)
        tsm = lmdb.open(tsm_path, readonly=True, lock=False)


    splits = ['train', 'validation']
    for split in splits:

        # Create output directory
        output_path = os.path.join(args.output, split)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        annotations_path = os.path.join(args.path_to_metadata, f'{split}.csv')
        annotations = pandas.read_csv(annotations_path, header=0, low_memory=False, names=columns)
        all_frames_dict = dict() # all_frames_dict is a dictionary keeping a list of segments against every video id

        # Initiate empty list for each video id in  all_frames_dict
        for kli in range(len(jsons_list_poses)):
            all_frames_dict[jsons_list_poses[kli]] = []

        gather_split_annotations(annotations, all_frames_dict, jsons_list_poses)

        # Remove unwanted data
        #removed_counter = 0
        #for key, data in list(all_frames_dict):
        #    if not args.keep_zero_shot:
        #        if not data['shared']:
        #            del all_frames_dict[key]
        #            removed_counter += 1
        #
        #print(f'removed unwanted data: {removed_counter}')

        for klk in range(len(jsons_list_poses)):
            video_id = jsons_list_poses[klk].strip().split('.')[0]

            # Get the list of segments for each video
            all_segments = all_frames_dict[jsons_list_poses[klk]]
            if len(all_segments) == 0:
                continue

            # Sort the segments based on start_frame
            all_segments = sorted(all_segments, key=itemgetter('start_frame'))

            # Read handpose file for the video and get list of frames with handposes for them
            poses_path = os.path.join(args.path_to_poses, jsons_list_poses[klk])
            with open(poses_path) as f:
                poses = json.load(f)

            # Convert json file to numpy array
            hands = []
            for hand in range(0, 2):
                hands.append(
                    numpy.stack([numpy.array(poses[i]['landmarks'][str(hand)], dtype='float32') for i in range(len(poses))]))
    
            # NOTE: Change framerate to 30 fps
            poses_data = numpy.stack(hands)
            poses_data = poses_data[:, ::2, :, :]
            poses_data = einops.rearrange(poses_data, 'H T ... -> T H ...')
            frames_count = poses_data.shape[0]

            # Store also frame embeddings
            embeddings = numpy.zeros((frames_count, 2048))
            database_frames_count = 0

            complete = True
            with tsm.begin(write=False) as e:
                for i in range(0, frames_count):
                    # example: 'nusar-2021_action_both_9044-a08_9044_user_id_2021-02-19_083738/C10095_rgb/C10095_rgb_0000011685.jpg'
                    key = os.path.join(video_id, f'{args.view}/{args.view}_{(i+1):010d}.jpg')
                    frame_data = e.get(key.strip().encode('utf-8'))
                    if frame_data is None:
                        print(f"[!] No data found for key={key}.")
                        complete = False
                        break

                    embeddings[i, :] = numpy.frombuffer(frame_data, 'float32')
                    database_frames_count += 1

            # Skip sample is frames are missing
            if not complete:
                # If only the ending frames are missing, continue
                if frames_count - database_frames_count < 100:
                    embeddings = embeddings[:database_frames_count, ...]
                    poses_data = poses_data[:database_frames_count, ...]
                    frames_count = database_frames_count
                else:
                    print(f'[!] missing frames for {video_id}, skipping sample')
                    continue

            # Generate fine labels
            labels = numpy.zeros((frames_count, 3))
            for segment in all_segments:
                beg, end = segment['start_frame'], segment['end_frame']
                labels[beg:end, 0] = int(segment['action']) + 1
                labels[beg:end, 1] = int(segment['verb'])   + 1
                labels[beg:end, 2] = int(segment['noun'])   + 1

            result = {
                'video_id': video_id,
                'view': args.view,
                'fine-labels': labels,
                'embeddings': embeddings,
                'poses': poses_data,
            }

            # Save complete sample in the output folder
            result_path = os.path.join(output_path, video_id + '.pkl')
            with open(result_path, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_metadata', type=str, default='data/Assembly101/metadata')
    parser.add_argument('--path_to_poses',    type=str, default='data/Assembly101/poses@60fps')
    parser.add_argument('--path_to_tsm',      type=str, default='data/Assembly101/TSM')
    parser.add_argument('--output',           type=str, default='data/Assembly101/processed')
    
    # What view to consider
    parser.add_argument('--view', type=str, default='C10095_rgb')
    parser.add_argument('--egocentric', action='store_true')

    # Keep zero-shot elements
    parser.add_argument('--keep_zero_shot', action='store_true')

    args = parser.parse_args()
    preprocess(args)