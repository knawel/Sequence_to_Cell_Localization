from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz",
    'sequence_max_length': 1500
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v6' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 17,
    'batch_size': 128,
    'log_step': 1024,
    'learning_rate': 1e-4,
    'hidden_size': 64,
    'layers': 2
}

# v1   -  length = 1500, small net
# v3   -  LSTM layers = 1
# v3   -  smaller LR
# v4   -  add softmax for output
# v5   - without softmax, just output


