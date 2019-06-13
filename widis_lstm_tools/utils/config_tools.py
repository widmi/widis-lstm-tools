# -*- coding: utf-8 -*-
"""config_tools.py: tools for parsing config files


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import os
import argparse
import json
import datetime
from collections import OrderedDict
from widis_lstm_tools.utils.collection import import_object
import torch
from copy import deepcopy


def get_config():
    """Reads json configfile
    
    Reads json configfile from path in first command line argument. If config file contains key 'results_dir', a
    directory with the current date will be created within the 'results_dir' path.
    
    Returns
    -------
    Configuration: ObjectDict
        ObjectDict object containing key value pairs from configuration file; Can be accessed like dictionary or object;
        
    results_dir: str
        Path to created results directory if config file contains key 'results_dir', otherwise None.
    
    Examples
    -------
    >>> config, resdir = get_config('path.json')
    >>> config.my_json_entry  # access like dictionary
    >>> config['my_json_entry']  # access like object
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', help='json configfile', type=str)
    args = parser.parse_args()
    configfile = args.configfile
    with open(configfile, 'r') as cf:
        config = ObjectDict(json.loads(cf.read()))
    try:
        resdir = os.path.join(config['results_dir'], datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(resdir, exist_ok=True)
    except KeyError:
        resdir = None
    return config, resdir


def layers_from_list(layerspecs, custom_types=None, verbose=False):
    """Create a torch.nn.Sequential() network OrderedDict from a list of layer specifications where layer classes are
    specified as strings
    
    Create a network from a list of layer specifications (dictionaries where layer classes are
    specified as strings); Layer in list will be created one-by-one, each with previous layer as input, as done by
    torch.nn.Sequential(); This allows to create a network from e.g. a json dictionary;
    
    Parameters
    ----------
    layerspecs : list of dict
        Network design as list of dictionaries, where each dict represents the layer parameters as kwargs and requires
        additional keys:
        layer : str
            A string value that is the layer class to instantiate.
            "layer" may include module paths (e.g. torch.nn.Conv2d()), modules will be imported automatically.
            "layer" may also be a key to a class in custom_types.
        name : str
            Name of the layer. Will serve as key to to the layer in the torch.nn.Sequential() OrderedDict().
    custom_types : dict
        Dictionary containing custom layer classes you want to use (if you do not want to import the class).
        E.g. custom_types={'my_layer':my_layer_class} will make the class my_layer_class available via layerspecs entry
        {"type": "my_layer", "name": "my_layer1"}.
    
    Returns
    ----------
    network_dict : torch.nn.Sequential() network OrderedDict
        Network as list of Pytorch layers in torch.nn.Sequential() OrderedDict
        
    Example
    -------
    >>> layerspecs = [
    >>>   {"layer": "torch.nn.Conv2d", "name": "conv1", "in_channels": 32,  "out_channels": 16, "kernel_size": 3},
    >>>   {"layer": "torch.nn.ReLU", "name": "relu1"},
    >>>   {"layer": "torch.nn.Conv2d", "name": "conv2", "in_channels": 16,  "out_channels": 8, "kernel_size": 3},
    >>>   {"layer": "torch.nn.ReLU", "name": "relu2"}]
    >>> network = layers_from_specs(layerspecs=layerspecs)
    >>> # ...
    >>> x = torch.rand(size=(32, 55, 55), dtype=torch.float32)
    >>> output = network(x)
    """
    if custom_types is None:
        custom_types = {}
    
    layerspecs = deepcopy(layerspecs)
    layer_dict = OrderedDict()
    for layerspec in layerspecs:
        layertype = layerspec.pop('layer')
        orig_name = layerspec.pop('name')
        name = orig_name
        i = 1
        while name in layer_dict:
            name = f"{orig_name}_{i}"
            i += 1
        
        try:
            if layertype in custom_types:
                layer_instance = custom_types[layertype](**layerspec)
            else:
                layerclass = import_object(layertype)
                layer_instance = layerclass(**layerspec)
        except Exception as e:
            print(f"Error in layer {layerclass} ({name}) with kwargs {layerspec}")
            raise e
        
        layer_dict[name] = layer_instance
    
    if verbose:
        print(f"Creating network:\n{layer_dict}")
    
    return torch.nn.Sequential(layer_dict)


class ObjectDict(OrderedDict):
    """ OrderedDict that also allows for object access (modified from https://goodcode.io/articles/python-dict-object/)
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
