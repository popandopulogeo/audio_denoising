import argparse
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from collections import OrderedDict
from os.path import join, basename, normpath, exists
from os import remove, listdir, pardir
from glob import glob
import librosa
import math
import json
from tqdm import tqdm
from common import Config, PHASE_TESTING, PHASE_TRAINING

AUDIO_EXT = '.wav'

JSON_DUMP_PARAMS = dict(indent=4, sort_keys=False, ensure_ascii=False, separators=(',', ':'))

def process_audio(dataset_dir, file_id, json_file, ext=AUDIO_EXT):
    """Process an audio file, or all the audio files with the same audio_id
    
    Arguments:
        dataset_dir {str} -- Path to the dir
        file_id {str} -- Audio id of the to-be-processed audio file
        json {str} -- Path to the json file containing detailed information about audio file
    
    Keyword Arguments:
        ext {str} -- Audio extension (default: {AUDIO_EXT})
    
    Returns:
        int -- Return the number of failed audio files
    """

    json_file = join(dataset_dir, json_file)

    hierarchy = OrderedDict([('dataset_path', dataset_dir)])

    file_info = OrderedDict()
    file_info['path'] = join(dataset_dir, file_id+ext)
    file_info['audio_sample_rate'] = librosa.get_samplerate(file_info['path'])
    file_info['duration'] = librosa.get_duration(filename=file_info['path'])
    file_info['audio_samples'] = math.ceil(file_info['audio_sample_rate'] * file_info['duration'])

    hierarchy['files'] = file_info
    json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)

    with open(json_file, 'wb') as f:
        f.write(json_str.encode())

def build_json_better(dataset_dir, dataset_csv, output_json, ext=AUDIO_EXT):
    """Build dataset json which contains detailed information about every audio file in the dataset
    
    Arguments:
        dataset_dir {str} -- Path to the dataset directory containing all the to-be-processed audio files
        dataset_csv {str} -- Path to the csv file describing the dataset
        output_json {str} -- Path to the final generated dataset json file
    
    Keyword Arguments:
        ext {str} -- Audio extension (default: {AUDIO_EXT})
        print_progress {bool} -- Print progress when True (default: {True})
    """

    files = pd.read_csv(dataset_csv, sep=',', header=None).iloc[:,0].to_list()

    print("Creating file's JSONs...")
    Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_audio)(dataset_dir, filename, filename + '.json', ext) for filename in tqdm(files))
    print("Complete!\n")

    combine_alljson(dataset_dir, output_json)


def build_csv(dataset_dir, output_csv, ext=AUDIO_EXT):
    with open(output_csv, 'w') as f:
        for path in Path(dataset_dir).rglob('*'+ext):
            f.write(str(path.stem) + '\n')

def combine_alljson(json_dir, output_json_path):
    """Combine all individual json files into a single json file with exclusion"""

    files = glob(join(json_dir,'*.json'))
    all_json_info = []

    print("JSONs combining...")

    for file in tqdm(files):
        with open(file, 'r') as fp:
            all_json_info.append(json.load(fp))
        remove(file)

    print("Complete!\n")

    json_clips = []
    for info in all_json_info:
        json_clips += [info['files']]

    json_clips = sorted(json_clips, key=lambda x:basename(x['path']))
    print('Num files: {}'.format(len(json_clips)))

    # build dictionary
    json_dict = OrderedDict()
    json_dict['dataset_path'] = normpath(join(all_json_info[0]['dataset_path'], pardir))
    json_dict['num_files'] = len(json_clips)
    json_dict['files'] = json_clips

    # write file
    print('Writing to "{}"...'.format(output_json_path))
    with open(output_json_path, 'wb') as f:
        json_str = json.dumps(json_dict, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())

    print('Complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, dest='phase', action='store', help="phase")
    parser.add_argument('--dataset', type=str, dest='dataset', action='store')
    parser.add_argument('--noisy',  default=False, action='store_true')
    parser.add_argument('--ext', type=str, dest='ext', default=AUDIO_EXT)
    args = parser.parse_args()

    config = Config()

    if args.phase in ['testing', 'training']:
        DIR = join(config.data_root, args.phase)
        CSV = join(config.data_root, args.phase, args.phase + config.csv_partial_name)
        JSON = join(config.data_root, args.phase + config.json_partial_name)

        if exists(CSV):
            remove(CSV)
        if exists(JSON):
            remove(JSON)

        build_csv(DIR, CSV, ext=args.ext)
        build_json_better(DIR, CSV, JSON, ext=args.ext)

    elif args.phase == 'asr':
        CLEAN_FILES = join(config.data_root, "clean_{}".format(args.dataset))
        if args.noisy:
            NOISY_FILES = join(config.data_root, "noisy_{}".format(args.dataset))

        TEXTS = join(config.data_root, "texts{}".format(args.dataset))
        CSV = join(config.data_root, args.phase + '_' + args.dataset + config.csv_partial_name)

        clean_audio = []
        if args.noisy:
            noisy_audio = []
        texts = []

        for path in tqdm(Path(CLEAN_FILES).rglob('*'+args.ext)):
            filename = str(path.stem)
            text_path = join(TEXTS,'{}.txt'.format(filename))
            with open(text_path, "r") as f_:
                text = f_.readline()
            noisy_audio.append(join(NOISY_FILES, "{}.{}".format(filename, args.ext)))
            if args.noisy:
                clean_audio.append(join(CLEAN_FILES, "{}.{}".format(filename, args.ext)))
            texts.append(text)

        if args.noisy:
            data = {'clean_audio':clean_audio,'noisy_audio':noisy_audio, 'text':texts}
        else:
            data = {'clean_audio':clean_audio,'text':texts}

        pd.DataFrame(data=data).to_csv(CSV)
