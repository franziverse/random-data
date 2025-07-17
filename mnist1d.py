""" 
This is copied (but slightly altered) from https://github.com/greydanus/mnist1d/tree/master/mnist1d
Avoids calling git in the code
"""
import pickle


def get_dataset_args(as_dict=False):
    """ Generate dictionary with dataset properties

    Parameters
    ----------
    as_dict : bool, optional
        if true, return the dataset properties as dictionary; if false, return an ObjectView, by default False

    Returns
    -------
    _type_
        _description_
    """
    arg_dict = {'num_samples': 5000,
            'train_split': 0.8,
            'template_len': 12,
            'padding': [36,60],
            'scale_coeff': .4, 
            'max_translation': 48,
            'corr_noise_scale': 0.25,
            'iid_noise_scale': 2e-2,
            'shear_scale': 0.75,
            'shuffle_seq': False,
            'final_seq_length': 40,
            'seed': 42,
            'url': 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'}
    return arg_dict


# args must not be a dict
def get_dataset(args, path=None, **kwargs):
    if 'args' in kwargs.keys() and kwargs['args'].shuffle_seq:
        shuffle = "_shuffle"
    else:
        shuffle = ""
    path = './mnist1d_data{}.pkl'.format(shuffle) if path is None else path

    try:
        dataset = from_pickle(path)
    except:
        raise FileNotFoundError(f"Required dataset file not found at: {path}")
    return dataset


def from_pickle(path): # load something
    value = None
    with open(path, 'rb') as handle:
        value = pickle.load(handle)
    return value