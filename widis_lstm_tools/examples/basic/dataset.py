# -*- coding: utf-8 -*-
"""dataset.py: Datasets


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import numpy as np
from torch.utils.data import Dataset


class RandomOrSigmoidal(Dataset):
    def __init__(self, n_samples: int, min_seq_len: int = 10, max_seq_len: int = 100):
        """Sequences need to be classified into random uniform signal or sigmoidal signal
        
        Dataset with sequences of different lengths containing a random uniform signal or a sigmoidal signal.
        
        Samples will be of shape (sequence, label, ID) with
        sequences: np.array
            Sequence of shape (seq_len, features)
        label: int
            Label (0=random signal, 1=sigmoidal signal)
        ID: str
            ID of sample
        """
        super(RandomOrSigmoidal, self).__init__()
        
        self.n_samples = int(n_samples)
        self.min_seq_len = int(min_seq_len)
        self.max_seq_len = int(max_seq_len)
        self.n_features = 1
        self.n_classes = 2
        
        self.labels = np.random.randint(low=0, high=2, size=(self.n_samples,))
        self.sequences = [self.__make_random__() if label == 0 else self.__make_sigmoidal__() for label in self.labels]
    
    def __make_sigmoidal__(self):
        start_end = np.random.uniform(low=0, high=np.pi*self.max_seq_len, size=(2,))
        start_end.sort()
        sequence = np.linspace(start=start_end[0], stop=start_end[1],
                               num=np.random.randint(low=self.min_seq_len, high=self.max_seq_len+1))
        np.sin(sequence, out=sequence)
        return sequence[..., None]
    
    def __make_random__(self):
        seq_len = np.random.randint(low=self.min_seq_len, high=self.max_seq_len+1)
        sequence = np.random.uniform(low=-1, high=1, size=(seq_len,))
        return sequence[..., None]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], str(idx)
