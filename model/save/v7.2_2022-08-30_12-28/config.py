from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz",
    'sequence_max_length': 512
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v7.2' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 20,
    'batch_size': 64,
    'log_step': 1024,
    'learning_rate': 1e-4,
    'hidden_size': 64,
    'layers': 2
}

# v7 finally works with long prots and CrossEntropyLoss
# v7.2 list of 3: membrane, cytosol and nuclear


