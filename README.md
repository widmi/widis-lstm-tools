# widis-lstm-tools v0.3

Various Tools for working with Long Short-Term Memory (LSTM) networks and sequences in [Pytorch](https://pytorch.org/),
aimed at getting your LSTM networks under control and providing a flexible template for your LSTM-needs.

LSTM paper (not earliest): https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735

## Quickstart

You can directly install the package from GitHub using the command below:

```bash
pip install git+https://github.com/widmi/widis-lstm-tools
```

To update your installation with or without dependencies, you can use:

```bash
pip install --upgrade git+https://github.com/widmi/widis-lstm-tools
```

or

```bash
pip install --no-dependencies --upgrade git+https://github.com/widmi/widis-lstm-tools
```

Run the simple example via

`python3 widis_lstm_tools/examples/basic/main.py config.json`

Run the more advanced example via

`python3 widis_lstm_tools/examples/model_selection/main.py config.json`

## Includes

- Flexible [LSTM Cell](widis_lstm_tools/nn.py#L67) and [LSTM Layer](widis_lstm_tools/nn.py#L394) implementation including
  - Easy access to individual forward and recurrent LSTM connections and biases, with options to cut/modify individual connections to gates or cell input (see e.g. https://arxiv.org/abs/1503.04069 for some useful modifications)
  - Plotting function for LSTM internal states
  - Support for Ticker/Tinker-Steps at the end of the sequence (https://arxiv.org/abs/1603.08983)
  - Automatic optimizations via [TorchScript](https://pytorch.org/docs/stable/jit.html) (loop optimization currently disabled due to gradient issues)
- Preprocessing tools
  - [Padding of different sequence lengths](widis_lstm_tools/preprocessing.py#L108)
  - [Encoding for nicely feeding float or integer numbers as inputs to the LSTM](widis_lstm_tools/preprocessing.py#L229)
- Other utilities
  - [Printing to log-file and console](widis_lstm_tools/utils/collection.py#L244)
  - [Splitting PyTorch datasets](widis_lstm_tools/preprocessing.py#L44), e.g. into training-, validation-, testset
  - [One-Hot encoding of n-dimensional arrays](widis_lstm_tools/preprocessing.py#L14)
  - [SaverLoader](widis_lstm_tools/collection.py#L56) class for saving/loading the most recent models to files or RAM objects
- Documented examples
  - [Basic LSTM example](widis_lstm_tools/examples/basic/main.py) with variable sequence lenghts
  - [Advanced LSTM example](widis_lstm_tools/examples/model_selection/main.py) with input encoding and model selection

## Requirements

- [Python3.6](https://www.python.org/) or higher
- Python packages:
   - [Pytorch](https://pytorch.org/) (tested with version 1.1.0)
   - [numpy](https://www.numpy.org/) (tested with version 1.16.2)
   - [matplotlib](https://matplotlib.org/) (tested with version 3.0.3)

<br/><br/>
I wish you the best for your work with LSTM!

-- Michael W. (widi)

