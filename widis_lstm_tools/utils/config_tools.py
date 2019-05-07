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
