import os
from tqdm import tqdm
import numpy

def output_model_predictions(path_to_output, model, dataloader, config):
    os.makedirs(path_to_output)

    with open(path_to_output + '/predictions.txt', 'wt') as f:

        # Header
        for category, num_classes in zip(config.categories, config.categories_num_classes):
            f.write(f'{category}:{num_classes}\n')

        f.write('================================\n')

        for frames, poses, targets in tqdm(dataloader, desc='saving model precitions'):
            assert(targets.shape[0] == 1)
            frames_count = targets.shape[1]

            f.write(f'{frames_count}\n')
            for category in range(targets.shape[-1]):
                cat_target = numpy.array(targets[0, :, category].cpu())
                for value in cat_target:
                    f.write(f'{value} ')
                f.write('\n')

            loss, outputs = model.predict_step((frames, poses, targets), None) 
            for category in range(targets.shape[-1]):
                cat_output = numpy.array(outputs[category][0].cpu())
                for value in cat_output:
                    f.write(f'{value} ')
                f.write('\n')

        f.flush()