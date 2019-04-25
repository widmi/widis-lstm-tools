# -*- coding: utf-8 -*-
"""nn.py: classes and functions for network architecture design and training


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import os
from collections import OrderedDict
from itertools import zip_longest
import time

import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt


def nograd(t):
    return t.detach()


class LearningRateDecay(object):
    def __init__(self, max_n_updates: int, optimizer, original_lr: float, final_lr=1e-5):
        """Decay learning rate linearly between original_lr and final_lr over course of learning
        
        Parameters
        ----------
        max_n_updates: int
            Maximum number of updates
        optimizer: PyTorch optimizer
            PyTorch optimizer, such as optim.Adam()
        original_lr: float
            Original learning rate
        final_lr: float
            Final learning rate
        
        Methods
        ----------
        get_lr: function
            Get current learning rate based on number of current update
        """
        self.max_n_updates = max_n_updates
        self.optimizer = optimizer
        self.original_lr = original_lr
        self.final_lr = final_lr
        
    def get_lr(self, update: int):
        """Get new decayed learning rate and set optimizer learning rate
        
        Parameters
        ----------
        update: int
            Number of current update
        
        Returns
        ----------
        new_lr: float
            Current learning rate
        """
        new_lr = self.original_lr * (1 - update / self.max_n_updates) + self.final_lr * (update / self.max_n_updates)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


class LSTMLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 W_ci=nn.init.normal_, W_ig=nn.init.normal_, W_og=nn.init.normal_, W_fg=False,
                 b_ci=nn.init.normal_, b_ig=nn.init.normal_, b_og=nn.init.normal_, b_fg=False,
                 a_ci=torch.tanh, a_ig=torch.sigmoid, a_og=torch.sigmoid, a_fg=lambda x: x, a_out=torch.tanh,
                 c_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 h_init=lambda t: nn.init.constant_(t, val=0).detach_(),
                 output_dropout_rate=0., return_all_seq_pos=False, n_tickersteps=0,
                 b_ci_tickers=nn.init.normal_, b_ig_tickers=nn.init.normal_, b_og_tickers=nn.init.normal_,
                 b_fg_tickers=False, inputformat='NLC'):
        """LSTM layer for different types of sequence predictions with inputs of shape [samples, sequence positions,
        features]

        Parameters
        -------
        in_features : int
            Number of input features
        out_features : int
            Number of output features (=number of LSTM blocks)
        W_ci, W_ig, W_og, W_fg : (list of) initializer functions or False
            Initial values or initializers for cell input, input gate, output gate, and forget gate weights; Can be list
            of 2 elements as [W_fwd, W_rec] to define different weight initializations for forward and recurrent
            connections respectively; If single element, forward and recurrent connections will use the same
            initializer/tensor; If set to False, connection will be cut;
            Shape of weights is W_fwd: [n_inputs, n_outputs], W_rec: [in_features, out_features];
        b_ci, b_ig, b_og, b_fg : initializer function or False
            Initial values or initializers for bias for cell input, input gate, output gate, and forget gate;
            If set to False, connection will be cut;
        a_ci, a_ig, a_og, a_fg, a_out : torch function
            Activation functions for cell input, input gate, output gate, forget gate, and LSTM output respectively;
        c_init : initializer function
            Initial values for cell states; Default: Zero and not trainable;
        h_init : initializer function
            Initial values for hidden states; Default: Zero and not trainable;
        output_dropout : float or False
            Dropout rate for LSTM output dropout (i.e. dropout of whole LSTM unit with rescaling of the remaining
            units); This also effects the recurrent connections;
        return_all_seq_pos : bool
            True: Return output for all sequence positions (continuous prediction);
            False: Only return output at last sequence position (single target prediction);
        n_tickersteps : int or False
            Number of ticker- or tinker-steps; n_tickersteps sequence positions without forward input will be added
            at the end of the sequence; During tickersteps, additional bias units will be added to the LSTM input;
            This allows the LSTM to perform computations after the sequence has ended;
        b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers : initializer function or False
            Initializers for bias applied during ticker steps for cell input, input gate, output gate, and forget gate;
            If set to False, connection will be cut;
        inputformat : 'NCL' or 'NLC'
            Input tensor format;
            'NCL' -> (batchsize, channels, seq.length);
            'NLC' -> (batchsize, seq.length, channels);
        
        """
        super(LSTMLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm_inlets = ['ci', 'ig', 'og', 'fg']
        self.n_tickersteps = n_tickersteps
        self.output_dropout_rate = output_dropout_rate
        self.return_all_seq_pos = return_all_seq_pos
        self.inputformat = inputformat
        
        # Get activation functions
        self.a = OrderedDict(zip(self.lstm_inlets + ['out'], [a_ci, a_ig, a_og, a_fg, a_out]))
        
        # Get initializer for tensors
        def try_split_w(w, i):
            try:
                return w[i]
            except TypeError:
                return w
        
        self.W_fwd_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(W, 0) for W in [W_ci, W_ig, W_og, W_fg]]))
        self.W_rec_init = OrderedDict(zip(self.lstm_inlets, [try_split_w(W, 1) for W in [W_ci, W_ig, W_og, W_fg]]))
        self.b_init = OrderedDict(zip(self.lstm_inlets, [b_ci, b_ig, b_og, b_fg]))
        self.b_tickers_init = OrderedDict(
            zip(self.lstm_inlets, [b_ci_tickers, b_ig_tickers, b_og_tickers, b_fg_tickers]))
        self.c_init = c_init
        self.h_init = h_init
        
        # Create parameters where needed and store them in dictionary for easier access
        self.W_fwd = OrderedDict(zip(self.lstm_inlets,
                                     [nn.Parameter(torch.FloatTensor(in_features, out_features))
                                      if self.W_fwd_init[i] is not False else False for i in self.lstm_inlets]))
        self.W_rec = OrderedDict(zip(self.lstm_inlets,
                                     [nn.Parameter(torch.FloatTensor(out_features, out_features))
                                      if self.W_rec_init[i] is not False else False for i in self.lstm_inlets]))
        self.b = OrderedDict(zip(self.lstm_inlets,
                                 [nn.Parameter(torch.FloatTensor(out_features))
                                  if self.b_init[i] is not False else False for i in self.lstm_inlets]))
        if self.n_tickersteps > 0:
            self.b_tickers = OrderedDict(
                zip(self.lstm_inlets, [nn.Parameter(torch.FloatTensor(out_features))
                                       if self.b_tickers_init[i] is not False else False for i in self.lstm_inlets]))
        else:
            self.b_tickers = False
        
        # Register parameters with module
        _ = [self.register_parameter('W_fwd_{}'.format(name), param) for name, param in self.W_fwd.items()
             if param is not False]
        _ = [self.register_parameter('W_rec_{}'.format(name), param) for name, param in self.W_rec.items()
             if param is not False]
        _ = [self.register_parameter('b_{}'.format(name), param) for name, param in self.b.items() if param
             is not False]
        if self.n_tickersteps > 0:
            _ = [self.register_parameter('b_tickers_{}'.format(name), param) for name, param in self.b_tickers.items()
                 if param is not False]
        
        self.c_first = nn.Parameter(torch.FloatTensor(out_features))
        self.h_first = nn.Parameter(torch.FloatTensor(out_features))
        
        self.output_dropout_mask = None
        
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = []
        self.h = []
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Initialize tensors
        self.reset_parameters()
        
        # This will optionally hold the true sequence lengths (before padding) for plotting
        self.true_seq_lens = None
        
        # This will hold the input sequences for plotting
        self.x = None
    
    def reset_parameters(self):
        
        # Apply initializer for W, b, and b_tickersteps
        _ = [self.W_fwd_init[i](self.W_fwd[i]) for i in self.lstm_inlets if self.W_fwd_init[i] is not False]
        _ = [self.W_rec_init[i](self.W_rec[i]) for i in self.lstm_inlets if self.W_rec_init[i] is not False]
        _ = [self.b_init[i](self.b[i]) for i in self.lstm_inlets if self.b_init[i] is not False]
        
        if self.n_tickersteps > 0:
            _ = [self.b_tickers_init[i](self.b_tickers[i]) for i in self.lstm_inlets
                 if self.b_tickers_init[i] is not False]
        
        # Initialize cell state and LSTM output at timestep -1
        self.c_init(self.c_first)
        self.h_init(self.h_first)
        
        # Manage LSTM output dropout (constant dropout of same units over time, including initial h state)
        if self.output_dropout_rate > 0:
            raise NotImplementedError("Sorry, LSTM output_dropout not available yet")
            self.output_dropout_mask = nograd(nn.init.uniform(nn.Parameter(torch.zeros_like(self.c_first),
                                                                           low=0, high=1)))
            self.a['out_dropout'] = lambda t: torch.where(self.output_dropout_mask > self.output_dropout_rate,
                                                          t, nograd(t * 0))
            self.h_first = self.a['out_dropout'](self.h_first)
    
    def reset_lstm_internals(self, n_batches):
        # Cell states and LSTM outputs at each timestep for each sample will be stored in a list
        self.c = [self.c_first.repeat((n_batches, 1))]
        self.h = [self.h_first.repeat((n_batches, 1))]
        
        # Activations of LSTM inlets at each timestep for each sample will be stored in a list
        self.lstm_inlets_activations = OrderedDict(zip(self.lstm_inlets, [[], [], [], []]))
        
        # Sample new output dropout mask
        if self.output_dropout_rate > 0:
            raise NotImplementedError("Sorry, LSTM output_dropout not available yet")
    
    def make_net_act(self, net_fwd, net_rec, b, a, n_samples, b_ticker=False):
        net_act = None
        if net_fwd is not False:
            net_act = net_fwd
        if net_rec is not False:
            if net_act is not None:
                net_act += net_rec
            else:
                net_act = net_rec
        if b is not False:
            if net_act is not None:
                net_act += b[None, :]
            else:
                net_act = b[None, :].repeat((n_samples, 1))
        if b_ticker is not False:
            if net_act is not None:
                net_act += b_ticker[None, :]
            else:
                net_act = b_ticker[None, :].repeat((n_samples, 1))
        if net_act is not None:
            net_act = a(net_act)
        else:
            net_act = 1
        return net_act
     
    def forward(self, x, true_seq_lens=None):
        """ Compute LSTM forward pass
        
        Compute LSTM forward pass.
        If using tickersteps and true_seq_lens, the tickerstep activations are appended at the end of the padded
        sequences. This also holds for the LSTM internals and the cellstate. However, the tickerstep activations are
        correctly computed beginning with the last activation at the true sequence length.
        
        Parameters
        ----------
        x: tensor
            Input tensor of shape (samples, length, channels) if NLC or (samples, channels, length) if NCL
        true_seq_lens: list or array of int
            Optional: True sequence lengths of padded sequences
            
        Returns
        --------
        h_out: tensor
            return_all_seq_pos == True: Output of LSTM for all (possibly padded) sequence positions in shape NLC or NCL.
            Tickerstep activations are appended at the end of the padded sequences if true_seq_lens is provided.
            return_all_seq_pos == False: Output of LSTM at last sequence position (after tickersteps) in shape NC.
        h: list of tensor
            List of LSTM outputs for every sequence position in shape list(NC)
        c: list of tensor
            List of LSTM cell states for every sequence position in shape list(NC)
        lstm_inlets_activations: OrderedDict of list of tensor
            OrderedDict of lists of internal LSTM states for every sequence position in shape list(NC) with keys
            ['ci', 'ig', 'og', 'fg'], representing cell input, input gate, output gate, and forget gate respectively.
        """
        self.true_seq_lens = true_seq_lens
        self.x = x
        
        if self.inputformat == 'NLC':
            n_batches, n_seqpos, n_features = x.shape
            seqpos_slice = lambda seqpos: [slice(None), seqpos]
        elif self.inputformat == 'NCL':
            n_batches, n_features, n_seqpos = x.shape
            seqpos_slice = lambda seqpos: [slice(None), slice(None), seqpos]
        else:
            raise ValueError("Input format {} not supported".format(self.inputformat))
        
        self.reset_lstm_internals(n_batches=n_batches)
        
        #
        # Calculate activations for every sequence position
        #
        for seq_pos in range(n_seqpos):
            # Calculate activations for LSTM inlets and append them to self.lstm_inlets_activations
            net_fwds = [torch.mm(x[seqpos_slice(seq_pos)], self.W_fwd[inlet]) if self.W_fwd[inlet] is not False
                        else False for inlet in self.lstm_inlets]
            net_fwds = OrderedDict(zip(self.lstm_inlets, net_fwds))
            net_recs = [torch.mm(self.h[-1], self.W_rec[inlet]) if self.W_rec[inlet] is not False
                        else False for inlet in self.lstm_inlets]
            net_recs = OrderedDict(zip(self.lstm_inlets, net_recs))
            
            net_acts = [self.make_net_act(net_fwds[inlet], net_recs[inlet], self.b[inlet], self.a[inlet], x.shape[0])
                        for inlet in self.lstm_inlets]
            _ = [self.lstm_inlets_activations[inlet].append(net_acts[i]) for i, inlet in enumerate(self.lstm_inlets)]
            
            # Calculate new cell state
            self.c.append(self.lstm_inlets_activations['ci'][-1] * self.lstm_inlets_activations['ig'][-1]
                          + self.c[-1] * self.lstm_inlets_activations['fg'][-1])
            
            # Calculate new LSTM output with new cell state
            if self.output_dropout_rate > 0:
                self.h.append(self.a['out_dropout'](self.a['out'](self.c[-1])) * self.lstm_inlets_activations['og'][-1])
            else:
                self.h.append(self.a['out'](self.c[-1]) * self.lstm_inlets_activations['og'][-1])
        
        #
        # Get LSTM output at sequence end, considering true_seq_len
        #
        if true_seq_lens is None:
            last_h = self.h[-1]
        else:
            last_h = torch.stack([self.h[tsl][sample_i] for sample_i, tsl in enumerate(true_seq_lens)], dim=0)
        
        #
        # Handle tickersteps
        #
        if self.n_tickersteps:
            #
            # Set input for tickersteps and consider true_seq_len
            #
            if true_seq_lens is None:
                ticker_h = [last_h]
                ticker_c = [self.c[-1]]
                ticker_lstm_inlets_activations = OrderedDict([(inlet, []) for inlet in self.lstm_inlets])
            else:
                ticker_h = [last_h]
                ticker_c = [torch.stack([self.c[tsl][sample_i] for sample_i, tsl in enumerate(true_seq_lens)], dim=0)]
                ticker_lstm_inlets_activations = OrderedDict([(inlet, []) for inlet in self.lstm_inlets])
            
            #
            # Calculate tickerstep activations
            #
            for _ in range(self.n_tickersteps):
                # Calculate activations for LSTM inlets and append them to ticker_lstm_inlets_activations, but during
                # tickersteps add tickerstep biases and set net_fwd to False
                net_recs = [torch.mm(ticker_h[-1], self.W_rec[inlet]) if self.W_rec[inlet] is not False
                            else False for inlet in self.lstm_inlets]
                net_recs = OrderedDict(zip(self.lstm_inlets, net_recs))
                net_acts = [self.make_net_act(False, net_recs[inlet], self.b[inlet], self.a[inlet], x.shape[0],
                                              self.b_tickers[inlet])
                            for inlet in self.lstm_inlets]
                _ = [ticker_lstm_inlets_activations[inlet].append(net_acts[i])
                     for i, inlet in enumerate(self.lstm_inlets)]
                
                # Calculate new cell state
                ticker_c.append(ticker_lstm_inlets_activations['ci'][-1] * ticker_lstm_inlets_activations['ig'][-1]
                                + ticker_c[-1] * ticker_lstm_inlets_activations['fg'][-1])
                
                # Calculate new LSTM output with new cell state
                if self.output_dropout_rate > 0:
                    ticker_h.append(self.a['out_dropout'](self.a['out'](ticker_c[-1]))
                                    * ticker_lstm_inlets_activations['og'][-1])
                else:
                    ticker_h.append(self.a['out'](ticker_c[-1]) * ticker_lstm_inlets_activations['og'][-1])
            
            # Remove initial step, as it is already in pre-tickerstep activations
            ticker_h = ticker_h[1:]
            ticker_c = ticker_c[1:]
            
            # Append tickerstep activations to end of (possibly padded) sequences
            self.h += ticker_h
            self.c += ticker_c
            _ = [self.lstm_inlets_activations[inlet].extend(ticker_lstm_inlets_activations[inlet])
                 for inlet in self.lstm_inlets]
            
            # Set output after tickersteps as new final output
            last_h = ticker_h[-1]
        
        #
        # Finalize output of LSTM
        #
        if self.return_all_seq_pos:
            # Output is LSTM output at each position
            if self.inputformat == 'NLC':
                h_out = torch.stack(self.h[1:], 1)
            elif self.inputformat == 'NCL':
                h_out = torch.stack(self.h[1:], 2)
            else:
                raise ValueError("Input format {} not supported".format(self.inputformat))
        else:
            # Output is LSTM output at last (existing) position
            h_out = last_h
        
        return h_out, self.h, self.c, self.lstm_inlets_activations
    
    def get_weights(self):
        """Return dictionaries for W_fwd and W_rec"""
        return self.W_fwd, self.W_rec
    
    def get_biases(self):
        """Return dictionaries for with biases and """
        return self.b
    
    def tensor_to_numpy(self, t):
        try:
            t = t.numpy()
        except TypeError:
            t = t.cpu().numpy()
        except AttributeError:
            t = np.as_array(t)
        except RuntimeError:
            t = t.clone().data.cpu().numpy()
        return t
    
    def plot_internals(self, mb_index: int = 0, filename: str = None, show_plot: bool = False, fdict: dict = None,
                       verbose=True):
        """Plot LSTM output, LSTM internal states, and LSTM input
        
        Plots LSTM input, LSTM output (h), LSTM cell state (c), LSTM gate activations, and LSTM cell input in separated
        axes. Each plot contains activations for all LSTM blocks (line colors identify individual LSTM units).
        
        Parameters
        ----------
        mb_index: int
            Index of sample to plot (index in minibatch)
        filename: str
            If specified, plot will be saved at location specified by filename. Necessary path will be created
            automatically.
        show_plot: bool
            True: Display plot
            False: Do not display plot
        fdict: dict
            kwargs passed to matplotlib.pyplot.subplots() function
        """
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if verbose:
            print(f"  Plotting LSTM states...", end='')
            start_time = time.time()
        
        #
        # Get sample activations at mb_index
        #
        lstm_internals_copy = OrderedDict([(k, self.tensor_to_numpy(torch.stack([t[mb_index] for t in v], 0)))
                                           for k, v in self.lstm_inlets_activations.items() if v[0] is not 1])
        lstm_internals_copy['c'] = self.tensor_to_numpy(torch.stack([t[mb_index] for t in self.c], 0))
        lstm_internals_copy['h'] = self.tensor_to_numpy(torch.stack([t[mb_index] for t in self.h], 0))
        lstm_internals_copy['input'] = self.tensor_to_numpy(self.x[mb_index])
        
        #
        # Remove activations between last true sequence position and start of tickersteps
        #
        if self.true_seq_lens is not None:
            true_seq_len = self.true_seq_lens[mb_index]
            for key in lstm_internals_copy.keys():
                if key == 'c' or key == 'h':
                    tsl = true_seq_len + 1
                else:
                    tsl = true_seq_len
                if key == 'input':
                    lstm_internals_copy[key] = lstm_internals_copy[key][:tsl]
                else:
                    lstm_internals_copy[key] = np.concatenate((lstm_internals_copy[key][:tsl],
                                                               lstm_internals_copy[key][-self.n_tickersteps:]), axis=0)
        
        plot_labels = [a for z in zip_longest(['input', 'h', 'c'], self.lstm_inlets) for a in z]
        max_len = max([v.shape[0] for v in lstm_internals_copy.values()])

        #
        # Do plotting: 1 axis per LSTM internal
        #
        if fdict is None:
            fdict = dict()
        # plt.rcParams.update({'font.size': 5})
        fig, axes = plt.subplots(4, 2, **fdict)
        axes = [a for aa in axes for a in aa]
        plt.tight_layout()
        for i, ax in enumerate(axes):
            try:
                label = plot_labels[i]
            except IndexError:
                ax.axis('off')
                continue
            ax.set_title(label)
            if label not in lstm_internals_copy.keys():
                continue
            
            _ = ax.plot(lstm_internals_copy[label], label=label)
            ax.set_xlim(left=0, right=max_len)
            ax.grid(True)
        
        if filename is not None:
            fig.savefig(filename)
        
        if show_plot:
            fig.show()
        else:
            plt.close(fig)
            del fig
        
        if verbose:
            print(f"    ...done! ({time.time() - start_time:8.7f}sec)")
