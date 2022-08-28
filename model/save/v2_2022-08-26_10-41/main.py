import torch as pt
import torch.nn as nn
from torch.utils.data import random_split
from src.dataset import SeqDataset
from torch.utils.data import DataLoader
from config import config_data, config_runtime


def train(config_data, config_runtime):
    from model import RNN
    from src.data_encoding import all_resnames, selected_locations
    _ = pt.manual_seed(42)
    pt.set_num_threads(12)
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    # read datasets
    nres = 1024
    dataset = SeqDataset(config_data['dataset_filepath'], nres_max=nres)
    print("length of the dataset is:", len(dataset))
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - 2502, 2502])
    print(f"Train: {len(train_dataset)}")
    print(f"Test: {len(test_dataset)}")

    n_letters = len(all_resnames)
    n_categories = len(selected_locations)
    learning_rate = config_runtime['learning_rate']
    n_hidden = config_runtime['hidden_size']
    n_layers = config_runtime['layers']

    model = RNN(nres, n_hidden, n_layers, n_categories, dev=device)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)

    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, Y) in enumerate(dataloader):
            # Compute prediction and loss
            x = X.to(device)
            y = Y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with pt.no_grad():
            for X, Y in dataloader:
                x = X.to(device)
                y = Y.to(device)
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
        #             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        #     correct /= size
        #     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(f"Avg loss: {test_loss:>8f} \n")

    # train model
    train_dataloader = DataLoader(train_dataset, batch_size=config_runtime['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config_runtime['batch_size'], shuffle=True)

    for t in range(config_runtime['num_epochs']):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    pt.save(model.state_dict(), "model.pt")


if __name__ == '__main__':
    # train model
    train(config_data, config_runtime)


