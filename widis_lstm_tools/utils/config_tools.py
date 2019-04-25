# -*- coding: utf-8 -*-
"""config_tools.py: tools for parsing config files


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import os
import argparse
import json
import datetime


def get_config():
    """Reads json configfile
    
    Reads json configfile from path in first command line argument. If config file contains key 'results_dir', a
    directory with the current date will be created within the 'results_dir' path.
    
    Returns
    -------
    Configuration: json dictionary
        Dictionary containing key value pairs from configuration file
    results_dir: str
        Path to created results directory if config file contains key 'results_dir', otherwise None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', help='json configfile', type=str)
    args = parser.parse_args()
    configfile = args.configfile
    with open(configfile, 'r') as cf:
        config = json.loads(cf.read())
    try:
        resdir = os.path.join(config['results_dir'], datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(resdir, exist_ok=True)
    except KeyError:
        resdir = None
    return config, resdir
