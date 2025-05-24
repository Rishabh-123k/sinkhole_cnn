import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import SinkholeCNN

def load_data(path):
    arr = np.load(path)
    sink = arr['sinkholes'].astype(np.float32)
    non = arr['non_sink'].astype(np.float32)
    X = np.concatenate([sink, non], axis=0)
    y = np.concatenate([
        np.ones(len(sink), dtype=np.float32),
        np.zeros(len(non), dtype=np.float32)
    ])
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(root, 'data', 'synthetic.npz')
    model_dir = os.path.join(root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, 'sinkhole_cnn_checkpoint.pth')
    final_path = os.path.join(model_dir, 'sinkhole_cnn.pth')

    batch_size = 32
    lr = 1e-3
    epochs_per_run = 20

    X, y = load_data(data_path)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds,   batch_size=batch_size)

    model = SinkholeCNN(input_length=X.shape[1])
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state'])
        opt.load_state_dict(ckpt['optim_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    end_epoch = start_epoch + epochs_per_run - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        for xb, yb in train_dl:
            preds = model(xb).squeeze()
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += loss_fn(model(xb).squeeze(), yb).item()
        val_loss /= len(val_dl)
        print(f"Epoch {epoch} validation loss {val_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': opt.state_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint for epoch {epoch}")

    torch.save(model.state_dict(), final_path)
    print(f"Training done. Final model saved to {final_path}")

if __name__ == '__main__':
    main()
