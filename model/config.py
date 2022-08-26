from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz"
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v2' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 16,
    'batch_size': 64,
    'log_step': 1024,
    'learning_rate': 1e-4,
    'hidden_size': 64,
    'layers': 2
}

# v1 - short list of locations
# v2 - bigger list of locations
