import os
import json
import shutil

def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'params.json')

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)

def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_parent_dir(path):
    """Get parent directory"""
    return os.path.abspath(os.path.join(path, os.pardir))

def get_path_same_dir(path, new_file_name):
    """Get the absolute path of a new file that is in the same directory as another file"""
    return os.path.join(get_parent_dir(path), new_file_name)

def print_dictionary(dictionary, sep=',', key_list=None, omit_keys=False):
    """ Pretty print dictionary """
    if key_list is None:
        key_list = dictionary.keys()
    print('{', end='')
    for idx, key in enumerate(key_list):
        end_str = sep + ' ' if idx < len(key_list)-1 else ''
        if not omit_keys:
            print('\'{}\': \'{}\''.format(key, dictionary[key]), end=end_str)
        else:
            print('\'{}\''.format(dictionary[key]), end=end_str)
    print('}')

def find_common_path(str1, str2, sep='/'):
    """ Find common path between two paths (strings) """
    return os.path.commonprefix([str1, str2]).rpartition(sep)[0]
