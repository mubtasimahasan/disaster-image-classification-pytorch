import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tabulate import tabulate
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def plot_loss_curves(results: Dict[str, List]):
    loss = [x.item() for x in results['train_loss']]
    valid_loss = [x.item() for x in results['valid_loss']]

    accuracy = results['train_acc']
    valid_accuracy = results['valid_acc']
    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, valid_loss, label='valid_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, valid_accuracy, label='valid_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def generate_confusion_matrix(test_results, class_names):
    y_true = test_results['targets']
    y_pred = test_results['predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def generate_classification_report(test_results, class_names):
    y_true = test_results['targets']
    y_pred = test_results['predictions']
    model_name = test_results['model_name']
    accuracy = test_results['accuracy']
    loss = test_results['loss']
    
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    return report


def merge_train_val(train_dataloader: DataLoader, valid_dataloader: DataLoader) -> DataLoader:
    merged_data = torch.utils.data.ConcatDataset([train_dataloader.dataset, valid_dataloader.dataset])
    merged_dataloader = DataLoader(merged_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
    return merged_dataloader


def cross_validation(merged_dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, num_folds: int = 5):
    kf = KFold(n_splits=num_folds)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(merged_dataloader.dataset)):
        print(f"Fold {fold + 1}/{num_folds}")
        train_subset = torch.utils.data.Subset(merged_dataloader.dataset, train_indices)
        val_subset = torch.utils.data.Subset(merged_dataloader.dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False)
        
        # Train the model
        model_instance = get_model(model)
        loss_fn= torch.nn.CrossEntropyLoss()
        optimizer_instance = torch.optim.SGD(params=model_instance.parameters(), lr=0.001)
        train_func(data_loader=train_loader, 
                   model=model_instance, 
                   loss_fn=loss_fn, 
                   optimizer=optimizer_instance, 
                   device=device)

        # Evaluate the model on validation data
        loaded_model = get_model(models.resnet50, pretrained=True, load=True)
        val_results = pred_func(data_loader = val_loader, 
                                model = loaded_model, 
                                loss_fn = loss_fn
                                ) 

        fold_accuracy = val_results["accuracy"]
        fold_f1_score = val_results["f1_score"]
        fold_results.append({"accuracy": fold_accuracy, "f1_score": fold_f1_score})

    return fold_results


def display_comparison_table_iv(test_results, class_names):
    classification_report = generate_classification_report(test_results, class_names)
    
    extracted_scores_dict = {}
    for class_name, metrics in classification_report.items():
        if class_name not in ['accuracy', 'weighted avg']:
            precision = metrics['precision']
            f1_score = metrics['f1-score']
            extracted_scores_dict[class_name] = {'precision': precision, 'f1_score': f1_score}  # Corrected line
    
    # Table IV from the paper
    table_iv = {
        'Damaged_Infrastructure': {'CAM-model': {'precision': 0.91, 'f1_score': 0.93}, 'TLAM-model': {'precision': 0.87, 'f1_score': 0.92}},
        'Fire_Disaster': {'CAM-model': {'precision': 1.00, 'f1_score': 0.98}, 'TLAM-model': {'precision': 1.00, 'f1_score': 1.00}},
        'Human_Damage': {'CAM-model': {'precision': 0.91, 'f1_score': 0.92}, 'TLAM-model': {'precision': 0.94, 'f1_score': 0.97}},
        'Water_Disaster': {'CAM-model': {'precision': 1.00, 'f1_score': 1.00}, 'TLAM-model': {'precision': 1.00, 'f1_score': 0.98}},
        'Land_Disaster': {'CAM-model': {'precision': 0.94, 'f1_score': 0.94}, 'TLAM-model': {'precision': 1.00, 'f1_score': 0.94}},
        'Non_Damage': {'CAM-model': {'precision': 0.97, 'f1_score': 0.96}, 'TLAM-model': {'precision': 1.00, 'f1_score': 0.99}},
        'macro avg': {'CAM-model': {'precision': 0.96, 'f1_score': 0.96}, 'TLAM-model': {'precision': 0.97, 'f1_score': 0.97}}
    }

    table_data = []
    for class_name, scores in extracted_scores_dict.items():
        cam_precision = f"{table_iv[class_name]['CAM-model']['precision']:.2f}"
        cam_f1_score = f"{table_iv[class_name]['CAM-model']['f1_score']:.2f}"
        tlam_precision = f"{table_iv[class_name]['TLAM-model']['precision']:.2f}"
        tlam_f1_score = f"{table_iv[class_name]['TLAM-model']['f1_score']:.2f}"
        extracted_precision = f"{scores['precision']:.2f}"
        extracted_f1_score = f"{scores['f1_score']:.2f}"
        table_data.append([class_name, cam_precision, cam_f1_score, tlam_precision, tlam_f1_score, extracted_precision, extracted_f1_score])

    headers = ["Class Names", "CAM Precision", "CAM F1", "TLAM Precision", "TLAM F1", "My Precision", "My F1"]

    print("Comparison of my result with dataset paper's TABLE IV: Performance Summary for CAM and TLAM on Test Data")
    print('Paper Link: https://arxiv.org/pdf/2107.01284v1.pdf')
    print('~' * 112)
    print(tabulate(table_data, headers=headers, tablefmt="pipe"))
    print('~' * 112)


def display_comparison_table_iii(fold_results: List[Dict[str, float]]):
    table_iii = {
        "CAM": {
            "Accuracy": [0.96, 0.96, 0.95, 0.96, 0.96],
            "F1 Score": [0.89, 0.90, 0.89, 0.92, 0.90]
        },
        "TLAM": {
            "Accuracy": [0.96, 0.96, 0.96, 0.97, 0.96],
            "F1 Score": [0.89, 0.88, 0.88, 0.92, 0.88]
        }
    }

    # Prepare the table
    print("Comparison of my fold-wise macro-average with paper's Table III: ross Validation Summary for CAM and TLAM")
    print('Paper Link: https://arxiv.org/pdf/2107.01284v1.pdf')
    
    print("+" * 80)
    print("| Fold | CAM Accuracy | CAM F1 | TLAM Accuracy | TLAM F1 | My Accuracy | My F1 |")
    print("|------|--------------|--------|---------------|---------|-------------|-------|")
    for fold in range(len(fold_results)):
        cam_accuracy = table_iii["CAM"]["Accuracy"][fold]
        cam_f1 = table_iii["CAM"]["F1 Score"][fold]
        tlam_accuracy = table_iii["TLAM"]["Accuracy"][fold]
        tlam_f1 = table_iii["TLAM"]["F1 Score"][fold]
        my_accuracy = fold_results[fold]["accuracy"] / 100
        my_f1 = fold_results[fold]["f1_score"]
        print(f"|  {fold+1}   |     {cam_accuracy:.2f}     |   {cam_f1:.2f} |      {tlam_accuracy:.2f}     |   {tlam_f1:.2f}  |     {my_accuracy:.2f}    |  {my_f1:.2f} |")
    print("+" * 80)

