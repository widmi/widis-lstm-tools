# -*- coding: utf-8 -*-
"""Example main file for LSTM network training

Scenario: LSTM network for predicting 1 label per sequence.

Input: Command line argument with path to config file 'config.json'.
Output: Output files will be saved in the output folder specified in 'config.json'.

Dataset: Dataset 'RandomOrSigmoidal' gives us sequences that need to be classified into random uniform signal or
sigmoidal signals.
Sequences have different lengths, so we need to use widis_lstm_tools.preprocessing.PadToEqualLengths for padding.
Config: Setup is done via config file 'config.json'.


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import os
import time
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from widis_lstm_tools.nn import LSTMLayer, LearningRateDecay
from widis_lstm_tools.utils.config_tools import get_config
from widis_lstm_tools.preprocessing import PadToEqualLengths
from widis_lstm_tools.examples.basic.dataset import RandomOrSigmoidal
from widis_lstm_tools.measures import bacc
from widis_lstm_tools.utils.collection import TeePrint, close_all


class Net(nn.Module):
    def __init__(self, n_input_features, n_lstm, n_outputs):
        super(Net, self).__init__()
        # Let's say we want an LSTM with forward connections to cell input and recurrent connections to input- and
        # output gate only; Furthermore we want a linear LSTM output activation instead of tanh:
        self.lstm1 = LSTMLayer(
            n_input_features, n_lstm,
            inputformat='NLC',  # Input format can be 'NLC' (samples, length, channels) or 'NCL'
            return_all_seq_pos=False,  # return predictions for last sequence positions only
            W_ci=(nn.init.normal_, False),  # cell input: weights to forward inputs (normal init)
            W_ig=(False, nn.init.normal_),  # input gate: weights to recurrent inputs (normal init)
            W_og=(False, nn.init.normal_,),  # output gate: weights to recurrent inputs (normal init)
            a_out=lambda x: x,  # LSTM output activation shall be identity function
            b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),  # neg. input gate bias for long seq
            n_tickersteps=5  # Optionally let LSTM do computations after sequence end, using tickersteps/tinkersteps
        )
        
        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = nn.Linear(n_lstm, n_outputs)
    
    def forward(self, x, true_seq_lens):
        # We only need the output of the LSTM; We get format (samples, n_lstm) since we set return_all_seq_pos=False:
        lstm_out, *_ = self.lstm1(x, true_seq_lens=true_seq_lens)
        net_out = self.fc_out(lstm_out)
        return net_out


def main():
    # Read config file path and set up results folder
    config, resdir = get_config()
    logfile = os.path.join(resdir, 'log.txt')
    os.makedirs(resdir, exist_ok=True)
    # Get a tprint() function that prints to stdout and our logfile
    tee_print = TeePrint(logfile)
    tprint = tee_print.tee_print
    
    # Set up PyTorch and set random seeds
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['rnd_seed'])
    np.random.seed(config['rnd_seed'])
    device = torch.device(config['device'])  # e.g. "cpu" or "cuda:0"
    
    # Get datasets
    trainset = RandomOrSigmoidal(n_samples=config['n_trainingset_samples'])
    testset = RandomOrSigmoidal(n_samples=config['n_testset_samples'])
    
    # Set up sequence padding
    padder = PadToEqualLengths(
        padding_dims=(0, None, None),  # only pad the first entry (sequences) in sample at dimension 0 (=seq.len.)
        padding_values=(0, None, None)  # pad with zeros
    )
    
    # Get Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2,
                                              collate_fn=padder.pad_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['batch_size'] * 4, shuffle=False,
                                             num_workers=2, collate_fn=padder.pad_collate_fn)

    # Create Network
    net = Net(n_input_features=trainset.n_features, n_lstm=config['n_lstm'], n_outputs=trainset.n_classes)
    net.to(device)

    # Get some loss functions
    mean_cross_entropy = nn.CrossEntropyLoss()
    
    # Get some optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # Get a linear learning rate decay
    lr_decay = LearningRateDecay(max_n_updates=config['n_updates'], optimizer=optimizer, original_lr=config['lr'])
    
    #
    # Start training
    #
    tprint("# settings: {}".format(config))
    update = 0
    while update < config['n_updates']:
        running_loss = 0.
        start_time = time.time()
        for data in trainloader:
            
            # Get and set current learning rate
            lr = lr_decay.get_lr(update)
            
            # Get samples
            inputs, labels, sample_id = data
            padded_sequences, seq_lens = inputs
            padded_sequences, labels = padded_sequences.to(device), labels.long().to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Get outputs for network
            outputs = net(padded_sequences, seq_lens)
            
            # Calculate loss, do backward pass, and update
            loss = mean_cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            update += 1
            
            # Update running losses for our statistic
            running_loss += loss.item() if hasattr(loss, 'item') else loss
            
            # Print current status and score
            if update % config['print_stats_at'] == 0 and update > 0:
                run_time = (time.time() - start_time) / config['print_stats_at']
                running_loss /= config['print_stats_at']
                tprint(f"[train] u: {update:07d}; loss: {running_loss:8.7f}; "
                       f"sec/update: {run_time:8.7f};lr: {lr:8.7f}")
                running_loss = 0.
                start_time = time.time()
            
            # Do some plotting using the LSTMLayer plotting function
            if update % config['plot_at'] == 0:
                # This will plot the LSTM internals for sample 0 in minibatch
                mb_index = 0
                pred = (outputs[mb_index, 1] > outputs[mb_index, 0]).float().cpu().item()
                net.lstm1.plot_internals(
                    filename=os.path.join(resdir, 'lstm_plots',
                                          f'u{update:07d}_id{sample_id[0]}_cl{labels[0]}_pr{pred}.png'),
                    mb_index=mb_index, fdict=dict(figsize=(50, 10), dpi=100))
                start_time = time.time()
        
            if update >= config['n_updates']:
                break

        print('Finished Training! Starting evaluation on test set...')
        
        # Compute scores on testset
        with torch.no_grad():
            tp_sum = 0.
            tn_sum = 0.
            p_sum = 0.
            loss = 0.
            for testdata in testloader:
                # Get samples
                inputs, labels, _ = testdata
                padded_sequences, seq_lens = inputs
                padded_sequences, labels = padded_sequences.to(device), labels.long().to(device)
                
                # Get outputs for network
                outputs = net(padded_sequences, seq_lens)
                
                # Add loss to mean loss over testset
                loss += (mean_cross_entropy(outputs, labels) * (len(labels) / len(testset)))
                
                # Store sum of tp, tn, t for BACC calculation
                labels = labels.float()
                p_sum += labels.sum(dim=0)  # number of positive samples
                predictions = (outputs[:, 1] > outputs[:, 0]).float()
                tp_sum += (predictions * labels).sum()
                tn_sum += ((1 - predictions) * (1 - labels)).sum()
            
            # Compute balanced accuracy
            n_sum = len(testset) - p_sum
            bacc_score = bacc(tp=tp_sum, tn=tn_sum, p=p_sum, n=n_sum).cpu().item()
            loss = loss.cpu().item()
            
            # Print results
            tprint(f"[eval] u: {update:07d}; loss: {loss:8.7f}; bacc: {bacc_score:5.4f}")
    
    print('Done!')


if __name__ == '__main__':
    try:
        main()
    finally:
        close_all()
