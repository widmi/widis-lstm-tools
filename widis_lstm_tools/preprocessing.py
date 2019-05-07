# -*- coding: utf-8 -*-
"""preprocessing.py: tools for data pre-processing


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def inds_to_one_hot(array: np.ndarray, n_features: int, new_dim: int = -1, return_dtype=np.float32):
    """Convert integer array to one-hot encoded array, assuming the maximum value is max_val; Add new feature dimension
    at new_dim;
    
    Parameters
    ----------
    array : np.ndarray
        (Possible multidimensional) integer indices of ones in one-hot array
    n_features : int
        Size of one-hot feature dimension, i.e. maximum index to expect for encoding
    new_dim : int
        Index of dimension to place new one-hot dimension at
    return_dtype : np.dtype
        Datatype of one-hot encoded array
    
    Returns
    ----------
    one_hot_array : np.ndarray
        One-hot encoded numpy array of datatype return_dtype
    """
    one_hot_array = np.zeros((array.size, n_features), dtype=return_dtype)
    one_hot_array[np.arange(array.size), array.flat] = 1.
    one_hot_array = one_hot_array.reshape((*array.shape, n_features))
    if new_dim != -1 and new_dim != len(array.shape):
        new_dim_order = list(range(len(array.shape)))
        new_dim_order = new_dim_order[:new_dim] + [len(array.shape)] + new_dim_order[new_dim:]
        one_hot_array = np.transpose(one_hot_array, axes=new_dim_order)
    return one_hot_array


def random_dataset_split(dataset: torch.utils.data.Dataset, split_sizes: tuple = (3 / 5., 1 / 5., 1 / 5.),
                         rnd_seed: int = 123, verbose: bool = True):
    """ Split a torch.utils.data.Dataset into multiple random splits, each provided via a dedicated
     torch.utils.data.Dataset instance.

     Randomly shuffly indices of dataset, then split dataset into splits according to split_sizes, and return a new
     torch.utils.data.Dataset instance for each split.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to split
    split_sizes : tuple
        Split sizes as fractions of the original dataset size (sum of split_sizes has to sum up to 1).
    rnd_seed : int
        Seed for random generator
    verbose : bool
        Verbose printing

    Returns
    ----------
    dataset_splits: list of DatasetSubset
        list of dataset splits as torch.utils.data.Dataset instances
    """
    original_dataset_len = dataset.__len__()
    original_indices = np.arange(original_dataset_len)
    rnd_gen = np.random.RandomState(rnd_seed)
    
    rnd_gen.shuffle(original_indices)
    
    split_inds_arrays = [original_indices[int(original_dataset_len * sum(split_sizes[:split_i])):
                                          int(original_dataset_len * sum(split_sizes[:split_i + 1]))]
                         for split_i in range(len(split_sizes))]
    if verbose:
        print(f"Split dataset into splits {split_sizes} "
              f"with {[len(s) for s in split_inds_arrays]} samples per split")
    
    split_datasets = [DatasetSubset(dataset, split_inds) for split_inds in split_inds_arrays]
    
    return split_datasets


class DatasetSubset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, subset_inds):
        """ Subset/split of a torch.utils.data.Dataset as new torch.utils.data.Dataset instance
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to take subset from
        subset_inds : list or array of inds
            list or np.array of integers with indices to use as subset
        """
        self.full_dataset = dataset
        self.subset_inds = subset_inds
    
    def __len__(self):
        return len(self.subset_inds)
    
    def __getitem__(self, idx):
        idx = self.subset_inds[idx]
        return self.full_dataset.__getitem__(idx)


class PadToEqualLengths(object):
    def __init__(self, padding_dims: tuple = (0,), padding_values: tuple = (0,)):
        """Pads minibatch entries to equal length based on the maximal length in the minibatch
        
        Provides a pad_collate_fn() method for usage with torch.utils.data.DataLoader() to pad minibatch sample entries
        to an equal length based on the maximal length in the current minibatch.
        Each minibatch consists of samples and each sample can have multiple entries (e.g. input and label).
        Samples are expected to be tuples or lists. Entries of samples can be arbitrary datatypes.
        padding_dims determines which entries should be padded at which dimension.
        Multiple entries per sample may be padded.
        Padded entries will be replaced by a tuple containing (padded_sequences, original_sequence_lengths).
        Entries that are not padded will be stacked to a tensor or returned as list if stacking is not possible.
        
        Parameters
        ----------
        padding_dims : tuple of integers
            Dimension index to pad for each entry in a sample. len(padding_dims) must be smaller or equal to the
            number of entries per sample. None values will not be padded. len(padding_dims) must be equal to
            len(padding_values).
           
        padding_values : tuple of values
            Values to pad with for each entry in a sample; len(padding_values) must be smaller or equal to the number
            of entries per sample. len(padding_dims) must be equal to len(padding_values).
        
        Example
        ----------
        > # Let's assume samples in the form of [input1, input2, labels]
        > # with inputs1 being arrays of shape (seq_length, features)
        > # and inputs2 being arrays of shape (x, y, features, seq_length)
        > # and labels as arrays of shape (n_classes).
        > # This will padd all samples to the same sequence length:
        > # import torch
        > # from widis-lstm-tools.preprocessing import PadToEqualLengths
        > seq_padding = PadToEqualLengths(padding_dims=(0, 3, None), padding_values=(0, 0, None))
        > data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=seq_padding.pad_collate_fn)
        > for data in data_loader:
        >   inputs1, inputs2, labels = data
        >   inputs1_sequences, inputs1_original_sequence_lengths = inputs1
        >   inputs2_sequences, inputs2_original_sequence_lengths = inputs2
        
        Provides
        ----------
        pad_collate_fn(batch_as_list)
            Function to be passed to torch.utils.data.DataLoader as collate_fn
        
        """
        if len(padding_dims) != len(padding_values):
            raise ValueError("padding_dims and padding_values must be of same length")
        
        self.padding_dims = padding_dims
        self.padding_values = padding_values
    
    def __pad_entries__(self, sequences: list, entry_ind: int):
        # If entry index should not be padded, stack entries to a tensor or return list of entries if not possible
        if entry_ind >= len(self.padding_dims) or self.padding_dims[entry_ind] is None:
            try:
                return torch.Tensor(np.array(sequences))
            except TypeError:
                return sequences
        
        # Get lengths of sequences
        seq_lens = torch.from_numpy(np.array([sequence.shape[self.padding_dims[entry_ind]] for sequence in sequences],
                                             dtype=np.int64))
        
        # Get maximum length and determine padded shape
        max_seq_len = seq_lens.max()
        new_shape = [len(sequences)] + list(sequences[0].shape)
        new_shape[self.padding_dims[entry_ind] + 1] = max_seq_len
        
        # Pre-allocate padded array
        try:
            torch_dtype = torch.from_numpy(np.array([], dtype=sequences[0].dtype)).dtype
            sequences = [torch.from_numpy(s) for s in sequences]
        except TypeError:
            torch_dtype = sequences[0].dtype
        if self.padding_values[entry_ind] == 0:
            padded_sequence_batch = torch.zeros(new_shape, dtype=torch_dtype)
        elif self.padding_values[entry_ind] == 1:
            padded_sequence_batch = torch.ones(new_shape, dtype=torch_dtype)
        else:
            padded_sequence_batch = torch.full(new_shape, dtype=torch_dtype, fill_value=self.padding_values[entry_ind])
        
        # Put sequences into padded array
        for i, sl in enumerate(seq_lens):
            dim_offset = [slice(0, None) for _ in range(self.padding_dims[entry_ind])]
            padded_sequence_batch[(i, *dim_offset, slice(0, sl))] = sequences[i]
        
        return padded_sequence_batch, seq_lens
    
    def pad_collate_fn(self, batch_as_list: list):
        """Function to be passed to torch.utils.data.DataLoader as collate_fn
        
        Function for usage with torch.utils.data.DataLoader() to pad minibatch samples to
        an equal length based on the maximal length in the current minibatch. Each minibatch consists of samples and
        each sample is expected to be a tuple or list. Multiple entries per sample may be padded. Padded entries
        will be replaced by a tuple containing (padded_sequences, original_sequence_lengths).
        
        Example
        ----------------
        > # Let's assume samples in the form of [ipnut1, input2, labels]
        > # with inputs1 being arrays of shape (seq_length, features)
        > # and inputs2 being arrays of shape (x, y, features, seq_length)
        > # and labels as arrays of shape (n_classes).
        > # This will padd all samples to the same sequence length:
        > # import torch
        > # from widis-lstm-tools.preprocessing import PadToEqualLengths
        > seq_padding = PadToEqualLengths(padding_dims=(0, 2, None), padding_values=(0, 0, None))
        > data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=seq_padding.pad_collate_fn)
        > for data in data_loader:
        >   inputs1, inputs2, labels = data
        >   inputs1_sequences, inputs1_original_sequence_lengths = inputs1
        >   inputs2_sequences, inputs2_original_sequence_lengths = inputs2
        """
        # Get number of entries per sample
        n_sample_entries = len(batch_as_list[0])
        # Padding all entries that are at the same position in all samples per batch
        padded_batch = [self.__pad_entries__([sample[entry_i] for sample in batch_as_list], entry_i)
                        for entry_i in range(n_sample_entries)]
        return padded_batch


class TriangularValueEncoding(object):
    def __init__(self, max_value, min_value, n_nodes: int, normalize: bool = False):
        """Encodes values in range [min_value, max_value] as array of shape (len(values), n_nodes)
        
        LSTM profits from having a numerical input with large range split into multiple input nodes; This class encodes
        a numerical input as n_nodes nodes with activations of range [0,1]; Each node represents a triangle of width
        triangle_span; These triangles are distributed equidistantly over the input value range such that 2 triangles
        overlap by 1/2 width and the whole input value range is covered; For each value to encode, the height of the
        triangle at this value is taken as node activation, i.e. max. 2 nodes have an activation > 0 for each input
        value, where both activations sum up to 1.
        
        Values are encoded via self.encode_value(value) and returned as float32 numpy array of length self.n_nodes;
        
        Parameters
        ----------
        max_value : float or int
            Maximum value to encode
        min_value : float or int
            Minimum value to encode
        n_nodes : int
            Number of nodes to use for encoding; n_nodes has to be larger than 1;
        normalize : bool
            Normalize encoded values? (default: False)
        """
        if n_nodes < 2:
            raise ValueError("n_nodes has to be > 1")
        
        # Set max_value to max value when starting from min value = 0
        max_value -= min_value
        
        # Calculate triangle_span (triangles overlap -> * 2)
        triangle_span = (max_value / (n_nodes - 1)) * 2
        
        self.n_nodes = int(n_nodes)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.triangle_span = float(triangle_span)
        self.normalize = normalize
    
    def encode_values(self, values):
        """Encode values as multiple triangle node activations

        Parameters
        ----------
        values : numpy.ndarray
            Values to encode as numpy array of shape (len(values),)
        
        Returns
        ----------
        float32 numpy array
            Encoded value as float32 numpy array of shape (len(values), n_nodes)
        """
        values = np.array(values, dtype=np.float)
        values[:] -= self.min_value
        values[:] *= ((self.n_nodes - 1) / self.max_value)
        encoding = np.zeros((len(values), self.n_nodes), np.float32)
        value_inds = np.arange(len(values))
        
        # low node
        node_ind = np.asarray(np.floor(values), dtype=np.int)
        node_activation = 1 - (values - node_ind)
        node_ind[:] = np.clip(node_ind, 0, self.n_nodes - 1)
        encoding[value_inds, node_ind] = node_activation
        
        # high node
        node_ind[:] += 1
        node_activation[:] = 1 - node_activation
        node_ind[:] = np.mod(node_ind, self.n_nodes)
        encoding[value_inds, node_ind] = node_activation
        
        # normalize encoding
        if self.normalize:
            encoding[:] -= (1 / self.n_nodes)
        
        return encoding
