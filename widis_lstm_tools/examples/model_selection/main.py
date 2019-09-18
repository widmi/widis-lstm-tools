# -*- coding: utf-8 -*-
"""Example main file for LSTM network training using training-, validation-, and testset and input encoding

Scenario: LSTM network for predicting 1 label per sequence but this time we will tackle this task (more) proper:
We will split our dataset into training-, validation-, and testset via
widis_lstm_tools.preprocessing.random_dataset_split().
This allows us to train on the trainingset, determine the best model during training with the validation set
(="model selection"), and perform the evaluation of the best model after training on the testset.
Furthermore, we augment our dataset and encode the signal values in multiple input nodes via
widis_lstm_tools.preprocessing.TriangularValueEncoding(), which will make LSTM learning easier.

Input: Command line argument with path to config file 'config.json'.
Output: Output files will be saved in the output folder specified in 'config.json'.

Dataset: Dataset 'RandomOrSineEncoded' gives us sequences that need to be classified into random uniform signal or
sigmoidal signals.
Sequences have different lengths, so we need to use widis_lstm_tools.preprocessing.PadToEqualLengths for padding.
Values in sequences are encoded in 16 input features, so a sequence will have shape (seq_len, 16).
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
from widis_lstm_tools.preprocessing import PadToEqualLengths, random_dataset_split
from widis_lstm_tools.examples.model_selection.dataset import RandomOrSigmoidalEncoded
from widis_lstm_tools.measures import bacc
from widis_lstm_tools.utils.collection import TeePrint, SaverLoader, close_all


class Net(nn.Module):
    def __init__(self, n_input_features, n_lstm, n_outputs):
        super(Net, self).__init__()
        # Let's say we want an LSTM with forward connections to cell input and recurrent connections to input- and
        # output gate only; Furthermore we want a linear LSTM output activation instead of tanh:
        self.lstm1 = LSTMLayer(
            in_features=n_input_features, out_features=n_lstm,
            # Possible input formats: 'NLC' (samples, length, channels), 'NCL', or 'LNC'
            inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, nn.init.xavier_normal_),
            # output gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_og=(False, nn.init.xavier_normal_),
            # forget gate: disable all connection (=no forget gate) and bias
            w_fg=False, b_fg=False,
            # LSTM output activation shall be identity function
            a_out=lambda x: x,
            # Optionally use negative input gate bias for long sequences
            b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),
            # Optionally let LSTM do computations after sequence end, using tickersteps/tinkersteps
            n_tickersteps=5,
        )
        
        # This would be a fully connected LSTM (cell input and gates connected to forward and recurrent connections)
        # without tickersteps:
        # self.lstm1 = LSTMLayer(
        #     in_features=n_input_features, out_features=n_lstm,
        #     inputformat='NLC',
        #     w_ci=nn.init.xavier_normal_, b_ci=nn.init.normal_, # equal to w_ci=(nn.init.normal_, nn.init.normal_)
        #     w_ig=nn.init.xavier_normal_, b_ig=nn.init.normal_,
        #     w_og=nn.init.xavier_normal_, b_og=nn.init.normal_,
        #     w_fg=nn.init.xavier_normal_, b_fg=nn.init.normal_,
        #     a_out=lambda x: x
        # )
        
        # After the LSTM layer, we add a fully connected output layer
        self.fc_out = nn.Linear(n_lstm, n_outputs)
    
    def forward(self, x, true_seq_lens):
        # We only need the output of the LSTM; We get format (samples, n_lstm) since we set return_all_seq_pos=False:
        lstm_out, *_ = self.lstm1.forward(x,
                                          true_seq_lens=true_seq_lens,  # true sequence lengths of padded sequences
                                          return_all_seq_pos=False  # return predictions for last sequence position
                                          )
        net_out = self.fc_out(lstm_out)
        return net_out


def main():
    # Read config file path and set up results folder
    config, resdir = get_config()
    logfile = os.path.join(resdir, 'log.txt')
    checkpointdir = os.path.join(resdir, 'checkpoint')
    os.makedirs(resdir, exist_ok=True)
    os.makedirs(checkpointdir, exist_ok=True)
    tee_print = TeePrint(logfile)
    tprint = tee_print.tee_print
    
    # Set up PyTorch and set random seeds
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['rnd_seed'])
    np.random.seed(config['rnd_seed'])
    device = torch.device(config['device'])  # e.g. "cpu" or "cuda:0"
    
    # Get dataset and split it into training-, validation-, and testset
    full_dataset = RandomOrSigmoidalEncoded(n_samples=config['n_samples'])
    training_set, validation_set, test_set = random_dataset_split(dataset=full_dataset, split_sizes=(3/5., 1/5., 1/5.),
                                                                  rnd_seed=config['rnd_seed'])
    
    # Set up sequence padding
    padder = PadToEqualLengths(
        padding_dims=(0, None, None),  # only pad the first entry (sequences) in sample at dimension 0 (=seq.len.)
        padding_values=(0, None, None)  # pad with zeros
    )
    
    # Get Dataloaders
    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=config['batch_size'], 
                                                      shuffle=True,  num_workers=2, collate_fn=padder.pad_collate_fn)
    validation_set_loader = torch.utils.data.DataLoader(validation_set, batch_size=config['batch_size'] * 4,
                                                        shuffle=False, num_workers=2, collate_fn=padder.pad_collate_fn)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=config['batch_size'] * 4,
                                                  shuffle=False, num_workers=2, collate_fn=padder.pad_collate_fn)

    # Create Network
    net = Net(n_input_features=full_dataset.n_features, n_lstm=config['n_lstm'], n_outputs=full_dataset.n_classes)
    net.to(device)

    # Get some loss functions
    mean_cross_entropy = nn.CrossEntropyLoss()
    
    # Get some optimizer
    optimizer = optim.Adam(net.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # Get a linear learning rate decay
    lr_decay = LearningRateDecay(max_n_updates=config['n_updates'], optimizer=optimizer, original_lr=config['lr'])
    
    # Create a dictionary with objects we want to have saved and loaded if needed
    state = dict(net=net, optimizer=optimizer, update=0, best_validation_loss=np.inf, validation_bacc=0)
    
    # Use the SaverLoader class to save/load our dictionary to files or to RAM objects
    saver_loader = SaverLoader(save_dict=state, device=device, save_dir=checkpointdir, n_savefiles=1, n_inmem=1)
    
    def calc_score(scoring_dataloader, scoring_dataset):
        """Compute scores on dataset"""
        with torch.no_grad():
            tp_sum = 0.
            tn_sum = 0.
            p_sum = 0.
            avg_loss = 0.
            for scoring_data in scoring_dataloader:
                # Get samples
                inputs, labels, _ = scoring_data
                padded_sequences, seq_lens = inputs
                padded_sequences, labels = padded_sequences.to(device), labels.long().to(device)
            
                # Get outputs for network
                outputs = net(padded_sequences, seq_lens)
            
                # Add loss to mean loss over testset
                avg_loss += (mean_cross_entropy(outputs, labels) * (len(labels) / len(scoring_dataset)))
            
                # Store sum of tp, tn, t for BACC calculation
                labels = labels.float()
                p_sum += labels.sum()  # number of positive samples
                predictions = (outputs[:, 1] > outputs[:, 0]).float()
                tp_sum += (predictions * labels).sum()
                tn_sum += ((1 - predictions) * (1 - labels)).sum()
        
            # Compute balanced accuracy
            n_sum = len(scoring_dataset) - p_sum
            bacc_score = bacc(tp=tp_sum, tn=tn_sum, p=p_sum, n=n_sum).cpu().item()
            avg_loss = avg_loss.cpu().item()
        return bacc_score, avg_loss
    
    # Get current state for some Python variables
    update, best_validation_loss, validation_bacc = (state['update'], state['best_validation_loss'],
                                                     state['validation_bacc'])
    
    # Save initial model as first model
    saver_loader.save_to_ram(savename=str(update))
    
    #
    # Start training
    #
    tprint("# settings: {}".format(config))
    while update < config['n_updates']:
        running_loss = 0.
        start_time = time.time()
        for data in training_set_loader:
            
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
            
            if update % config['validate_at'] == 0 or update == config['n_updates']:
                # Calculate scores and loss on validation set
                print("  Calculating validation score...", end='')
                validation_start_time = time.time()
                validation_bacc, validation_loss = calc_score(scoring_dataloader=validation_set_loader,
                                                              scoring_dataset=validation_set)
                print(f" ...done! ({time.time()-validation_start_time})")
                tprint(f"[validation] u: {update:07d}; loss: {validation_loss:8.7f}; bacc: {validation_bacc:5.4f}")
                
                # If we have a new best loss on the validation set, we save the model as new best model
                if best_validation_loss > validation_loss:
                    best_validation_loss = validation_loss
                    tprint(f"  New best validation loss: {validation_loss}")
                    # Save current state as RAM object
                    state['update'], state['best_validation_loss'], state['validation_bacc'] = \
                        update, best_validation_loss, validation_bacc
                    saver_loader.save_to_ram(savename=str(update))
            
            if update >= config['n_updates']:
                break
        
    print('Finished Training! Starting evaluation on test set...')
    
    #
    # Now we have finished training and stored the best model in 'checkpoint/best.tar.gzip' and want to see the
    # performance of this best model on the test set
    #
    
    # Load best model from newest RAM object
    state = saver_loader.load_from_ram()
    update, best_validation_loss, validation_bacc = (state['update'], state['best_validation_loss'],
                                                     state['validation_bacc'])
    
    # Save model to file
    saver_loader.save_to_file(filename='best.tar.gzip')
    
    # Calculate scores and loss on test set
    print("  Calculating testset score...", end='')
    test_start_time = time.time()
    test_bacc, test_loss = calc_score(scoring_dataloader=test_set_loader, scoring_dataset=test_set)
    print(f" ...done! ({time.time()-test_start_time})")
    
    # Print results
    tprint(f"[test] u: {update:07d}; loss: {test_loss:8.7f}; bacc: {test_bacc:5.4f}")
    
    print('Done!')


if __name__ == '__main__':
    try:
        main()
    finally:
        close_all()
