from datetime import datetime

config_data = {
    'dataset_filepath': "data/data_seq_locations.xz",
    'sequence_max_length': 512
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v8.7' + tag,
    'output_dir': 'save',
    'device': 'cuda',
    'num_epochs': 38,
    'batch_size': 256,
    'log_step': 1024,
    'learning_rate': 5e-5,
    'hidden_size': 64,
    'layers': 1
}

# v7 finally works with long prots and CrossEntropyLoss
# 7.2 list of 3: membrane, cytosol and nuclear, works good
# 7.3, loss fn is BCEWithLogitsLoss
# 8 one layer of LSTM
# 8.1 bigger batch, smaller net
# 8.15 bigger LR
# 8.2 added vesicl
# 8.3 locations: ['membrane', 'nucle', 'golgi']
# 8.4 location ['membrane', 'nucle', 'extracell']
# 8.5 location ['membrane', 'nucle', 'cytoskelet'], slightly bigger model, smaller LR
# 8.6 all loc
# 8.7 loc=5
