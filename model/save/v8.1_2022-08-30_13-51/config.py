from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz",
    'sequence_max_length': 512
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v8.1' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 48,
    'batch_size': 256,
    'log_step': 1024,
    'learning_rate': 5e-5,
    'hidden_size': 32,
    'layers': 1
}

# v7 finally works with long prots and CrossEntropyLoss
# 7.2 list of 3: membrane, cytosol and nuclear, works good
# 7.3, loss fn is BCEWithLogitsLoss
# 8 one layer of LSTM
# 8.1 bigger batch, smaller net



