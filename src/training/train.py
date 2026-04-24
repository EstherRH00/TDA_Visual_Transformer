import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        if len(batch) == 3:
            x, tda, y = batch
            x, tda, y = x.to(device), tda.to(device), y.to(device)
            outputs = model(x, tda)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            outputs = model(x)

        loss = criterion(outputs.squeeze(), y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)