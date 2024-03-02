import torch
from torchvision import models

from models.model_training import get_model, train_and_evaluate, pred_func
from data.data_processing import get_data, display_random_images, get_dataloader
from utils.utility import plot_loss_curves, generate_confusion_matrix, display_comparison_table_iv, merge_train_val, cross_validation, display_comparison_table_iii
from utils.config import CFG

def main():
    # Set Seed
    torch.manual_seed(CFG.seed)
    # Load dataset from specified path
    dataset = get_data(CFG.path)

    # Display random images from the dataset
    display_random_images(dataset)

    # Get data loaders for training, testing, and validation
    train_dataloader, test_dataloader, valid_dataloader = get_dataloader(dataset)

    # Get ResNet model architecture with pretrained weights
    resnet_model = get_model(models.resnet50, pretrained=True)

    # Train and evaluate the ResNet model
    results = train_and_evaluate(train_dataloader=train_dataloader, 
                                valid_dataloader=valid_dataloader, 
                                model=resnet_model,
                                loss_fn=torch.nn.CrossEntropyLoss(),
                                optimizer=torch.optim.SGD(params=resnet_model.parameters(), lr=0.001),
                                device=CFG.device
                                )

    # Plot loss curves based on training and validation results
    plot_loss_curves(results)

    # Load a pretrained ResNet model
    loaded_model = get_model(models.resnet50, pretrained=True, load=True)

    # Generate predictions using the loaded model on the test dataset
    test_results = pred_func(data_loader=test_dataloader, 
                            model=loaded_model, 
                            loss_fn=torch.nn.CrossEntropyLoss(),
                            device=CFG.device
                            )   
    
    # Generate confusion matrix based on test results and defined classes
    generate_confusion_matrix(test_results, CFG.classes)

    # Display comparison table for test results
    display_comparison_table_iv(test_results, CFG.classes)

    # Merge train and  validation
    merged_dataloader = merge_train_val(train_dataloader=train_dataloader, 
                                    valid_dataloader=valid_dataloader
                                   )
    
    # Perform cross-validation and display comparison table for the results
    fold_results = cross_validation(merged_dataloader,
                                    model=models.resnet50,
                                    loss_fn=torch.nn.CrossEntropyLoss(),
                                    num_folds=5
                                    )

    display_comparison_table_iii(fold_results)

if __name__ == "__main__":
    main()