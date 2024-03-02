import torch
from torch.utils.data import DataLoader
from torchvision import  models
import torch.nn as nn
from typing import Tuple, Dict, List
from tqdm import tqdm 
from timeit import default_timer as timer
from sklearn.metrics import f1_score

from utils.config import CFG

def get_model(model: nn.Module, pretrained: bool = True, load: bool = False) -> nn.Module:
    # Load the pre-trained model
    model_instance = model(pretrained=pretrained)
    
    if isinstance(model_instance, models.resnet.ResNet):
        num_ftrs = model_instance.fc.in_features
        model_instance.fc = nn.Linear(num_ftrs, len(CFG.classes)).to(CFG.device)
    else:
        raise NotImplementedError("Model type not supported for automatic modification of classifier layer.")
    
    if load:
        model_instance.load_state_dict(torch.load(f="./model_1.pth"))
    
    return model_instance


def train_func(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: None):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def validation_func(data_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: None):
    valid_loss, valid_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            valid_pred = model(X)
            valid_loss += loss_fn(valid_pred, y)
            valid_acc += accuracy_fn(y_true=y, y_pred=valid_pred.argmax(dim=1))
        valid_loss /= len(data_loader)
        valid_acc /= len(data_loader)
        return valid_loss, valid_acc


def pred_func(data_loader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: None):
    eval_loss, eval_acc = 0, 0
    model.to(device)
    model.eval()
    y_preds = []
    y_targets = []
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader)):
            X, y = X.to(device), y.to(device)
            eval_pred = model(X)
            eval_loss += loss_fn(eval_pred, y)
            eval_acc += accuracy_fn(y_true=y, y_pred=eval_pred.argmax(dim=1))
            eval_labels = torch.argmax(torch.softmax(eval_pred, dim=1), dim=1)
            y_preds.append(eval_labels)
            y_targets.append(y)

        eval_loss /= len(data_loader)
        eval_acc /= len(data_loader)

        y_preds = torch.cat(y_preds).cpu() 
        y_targets = torch.cat(y_targets).cpu() 

        f1 = f1_score(y_targets, y_preds, average='macro')

        return {"model_name": model.__class__.__name__, 
                "loss": eval_loss.item(), 
                "accuracy": eval_acc, 
                "f1_score": f1,
                "predictions": y_preds, 
                "targets": y_targets}


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc


def print_train_time(start: float, end: float, device: CFG.device):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_and_evaluate(train_dataloader: torch.utils.data.DataLoader,
                       valid_dataloader: torch.utils.data.DataLoader,
                       model: torch.nn.Module,
                       loss_fn: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       device: None):
    
    class_names = CFG.classes
    train_time_start_model_1 = timer()
    best_loss = 10
    results = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    for epoch in tqdm(range(CFG.num_epochs)):
        train_loss, train_acc = train_func(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        valid_loss, valid_acc = validation_func(data_loader=valid_dataloader, model=model, loss_fn=loss_fn, device=device)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.2f} | Test loss: {valid_loss:.3f} | Test acc: {valid_acc:.2f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)
        if valid_loss < best_loss:
            best_loss = valid_loss
            print(f"Saving best model for epoch: {epoch}")
            torch.save(obj=model.state_dict(), f="./model_1.pth")
    train_time_end_model_1 = timer()
    total_train_time_model_1 = print_train_time(start=train_time_start_model_1, end=train_time_end_model_1, device=device)
    return results

