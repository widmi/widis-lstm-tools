# -*- coding: utf-8 -*-
"""collection.py: miscellaneous


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import os
import sys
import copy
import numpy as np
import torch
import importlib

from collections import OrderedDict

FILEHANDLES = []


def import_object(objname):
    objmodule = importlib.import_module(objname.split('.', maxsplit=1)[0])
    return get_rec_attr(objmodule, objname.split('.', maxsplit=1)[-1])


def get_rec_attr(module, attrstr):
    """Get attributes and do so recursively if needed"""
    if attrstr is None:
        return None
    attrs = attrstr.split('.')
    for attr in attrs:
        module = getattr(module, attr)
    return module


def close_all():
    """Try to flush and close all file handles"""
    global FILEHANDLES
    for fh in FILEHANDLES:
        try:
            fh.flush()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass


def int_to_one_hot(integers, max_int):
    one_hot = np.zeros((len(integers), max_int), dtype=np.int)
    one_hot[np.arange(len(integers)), integers] = 1
    return one_hot


class SaverLoader(object):
    def __init__(self, save_dict: dict, device: str, save_dir: str = '', n_savefiles: int = 1, n_inmem: int = 1):
        """Save/load PyTorch models and Python objects to/from memory or files, keeping the last n_savefiles files and
        n_inmem memory saves.
        
        Parameters
        ----------
        save_dict: dict
            Dictionary with objects to save, may contain PyTorch models that support the load_state_dict() function or
            other Python objects. This dictionary will be modified in-place when loading a state.
        device: str
            PyTorch device to load saved states to (e.g. "cpu" or "cuda:0")
        save_dir: str
            Parent directory to place saved files in
        n_savefiles: int
            Number of saved files to keep. If this number is exceeded, the oldest saved file will be removed.
            Set to 0 to keep all saved states.
        n_inmem: int
            Number of saved states in RAM to keep. If this number is exceeded, the oldest saved state will be removed.
            Set to 0 to keep all saved states.
            
        Methods
        ----------
        save_to_file
            Save state to a file
        save_to_ram
            Save state to RAM object
        load_from_file
            Load state from file
        load_from_ram
            Load state from RAM object
        get_saved_to_file
            Get names of saved files
        get_saved_to_ram
            Get names of saved RAM objects
        """
        self.save_dir = save_dir
        self.save_dict = save_dict
        self.n_savefiles = n_savefiles
        self.saved_to_file = OrderedDict()
        self.saved_to_ram = OrderedDict()
        self.n_inmem = n_inmem
        self.device = device
        os.makedirs(save_dir, exist_ok=True)
    
    def save_to_file(self, filename: str, separate_save: bool = False, verbose: bool = True):
        """Save state to a file filename
        
        Save state to a file filename in directory self.save_dir. Only keep last self.n_savefiles saved files, unless
        separate_save==True, in which case the saved state will not be tracked.
        
        Parameters
        ----------
        filename : str
            Name or path of file to save to in parent directory self.save_dir
        separate_save : bool
            If True, the saved file will be tracked and only the last self.n_savefiles saved files will be kept.
            If False, the file will not be tracked.
        verbose : bool
            Verbose output
        """
        if verbose:
            print('  Saving checkpoint to file...', end='')
        savepath = os.path.join(self.save_dir, filename)
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        save_dict = dict([(k, self.save_dict[k]) if not hasattr(self.save_dict[k], 'state_dict') else
                          (k, self.save_dict[k].state_dict())
                          for k in self.save_dict.keys()])
        
        torch.save({**save_dict}, savepath)
        if not separate_save:
            self.saved_to_file[savepath] = savepath
            if self.n_savefiles != 0 and (len(self.saved_to_file) > self.n_savefiles):
                rem_key = list(self.saved_to_file.keys())[0]
                os.remove(rem_key)
                del self.saved_to_file[rem_key]
        if verbose:
            print(' done!')
        return savepath
    
    def save_to_ram(self, savename: str, verbose: bool = True):
        """Save state to a RAM object
        
        Save state to a RAM object. Only keep last self.n_inmem saved states.
        
        Parameters
        ----------
        savename : str
            Object-name to use as key to the RAM object
        verbose : bool
            Verbose output
        """
        if verbose:
            print('  Saving checkpoint to memory...', end='')

        save_dict = dict([(k, self.save_dict[k]) if not hasattr(self.save_dict[k], 'state_dict') else
                          (k, self.save_dict[k].state_dict())
                          for k in self.save_dict.keys()])
        
        save_dict = copy.deepcopy(save_dict)
        self.saved_to_ram[savename] = save_dict
        if self.n_inmem != 0 and (len(self.saved_to_ram) > self.n_inmem):
            rem_key = list(self.saved_to_ram.keys())[0]
            del self.saved_to_ram[rem_key]
        if verbose:
            print(' done!')
    
    def load_from_file(self, loadname: str = None, verbose: bool = True):
        """Load state from saved file
        
        Load state from a file loadname in directory self.save_dir, from an external file, or load newest saved file.
        
        Parameters
        ----------
        loadname : str
            Name or path of file to load from.
            Can be key to previously saved file (see get_saved_to_file()), or external filepath.
            If None, the newest saved file will be loaded.
        verbose : bool
            Verbose output
        """
        if loadname is None:
            loadname = list(self.saved_to_file.keys())[-1]
            
        if verbose:
            print(f'  Loading checkpoint from file "{loadname}"...', end='')
        
        try:
            load_dict = torch.load(self.saved_to_file[loadname])
        except KeyError:
            load_dict = torch.load(loadname)
        
        for k in self.save_dict.keys():
            if k not in load_dict.keys():
                print(f"Warning: Could not load {k}!")
            elif hasattr(self.save_dict[k], 'load_state_dict'):
                self.save_dict[k].load_state_dict(load_dict[k])
                if hasattr(self.save_dict[k], 'to'):
                    self.save_dict[k].to(self.device)
            else:
                self.save_dict[k] = copy.deepcopy(load_dict[k])
        if verbose:
            print(' done!')
        return self.save_dict
    
    def load_from_ram(self, loadname: str = None, verbose: bool = True):
        """Load state from RAM object
        
        Load state from a RAM object with name loadname or load newest saved file.
        
        Parameters
        ----------
        loadname : str
            Name of RAM object to load.
            If None, the newest saved RAM object will be loaded.
        verbose : bool
            Verbose output
        
        Returns
        -------
        self.save_dict : dict
            Loaded dictionary (original self.save_dict is also modified in-place)
        """
        if loadname is None:
            loadname = list(self.saved_to_ram.keys())[-1]
            
        if verbose:
            print(f'  Loading checkpoint from memory "{loadname}"...', end='')
        
        load_dict = self.saved_to_ram[loadname]
        for k in self.save_dict.keys():
            if k not in load_dict.keys():
                print(f"Warning: Could not load {k}!")
            elif hasattr(self.save_dict[k], 'load_state_dict'):
                self.save_dict[k].load_state_dict(load_dict[k])
                if hasattr(self.save_dict[k], 'to'):
                    self.save_dict[k].to(self.device)
            else:
                self.save_dict[k] = copy.deepcopy(load_dict[k])
        if verbose:
            print(' done!')
        return self.save_dict
    
    def get_saved_to_file(self):
        """Get names of saved files"""
        return list(self.saved_to_file.keys()).copy()
    
    def get_saved_to_ram(self):
        """Get names of saved RAM objects"""
        return list(self.saved_to_ram.keys()).copy()


class TeePrint(object):
    def __init__(self, *filepaths, overwrite_existing: bool = False):
        """ Provides .tee_print() method to print to stdout and list of filepaths
        
        Parameters
        ----------
        filepaths: str
            Filepaths of files to print to
        overwrite_existing : bool
            True: overwrite existing files in filepaths
            False: append to existing files in filepaths
        
        Methods
        ----------
        tee_print(*args, **kwargs)
            Print to stdout and files (parameters are fed to print(*args, **kwargs))
        file_print(*args, **kwargs)
            Print to files only (parameters are fed to print(*args, **kwargs))
        flush()
            Flush stdout and filehandles
        close()
            Close filehandles
        """
        self.filepaths = filepaths
        self.original_stdout = sys.stdout
        if overwrite_existing:
            self.filehandles = [open(filepath, 'w') for filepath in filepaths]
        else:
            self.filehandles = [open(filepath, 'a') for filepath in filepaths]
        
        global FILEHANDLES
        FILEHANDLES += self.filehandles
    
    def tee_print(self, *args, **kwargs):
        print(*args, **kwargs)
        _ = [print(*args, **kwargs, file=fh) for fh in self.filehandles]
    
    def file_print(self, *args, **kwargs):
        _ = [print(*args, **kwargs, file=fh) for fh in self.filehandles]
    
    def flush(self):
        for fh in self.filehandles + [self.original_stdout]:
            try:
                fh.flush()
            except Exception:
                pass
    
    def close(self):
        for fh in self.filehandles:
            try:
                fh.close()
            except Exception:
                pass