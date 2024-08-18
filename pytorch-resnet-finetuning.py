import torch

import matplotlib.pyplot as plt

def train(
    train_loader,  # DataLoader for the training dataset
    val_loader,  # DataLoader for the validation dataset
    model,  # PyTorch model to be trained
    loss_fn,  # Loss function used for training
    optimizer,  # Optimizer used for training
    scheduler,  # Learning rate scheduler
    device,  # Device on which the model and data will be loaded
    epochs,  # Number of training epochs
    lr,  # Learning rate for the optimizer
    step_size,  # Step size for the learning rate scheduler
    save_path,  # File path to save the best model state
):
    """
    Trains a PyTorch model using the provided data loaders, loss function, optimizer, and scheduler.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        model (torch.nn.Module): PyTorch model to be trained.
        loss_fn (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device on which the model and data will be loaded.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        step_size (int): Step size for the learning rate scheduler.
        save_path (str): File path to save the best model state.

    Returns:
        dict: A dictionary containing the training history, including train and validation accuracy and loss.
        str: File path of the saved image.
    """
    model = model.to(device)

    # Define loss function, optimizer, and scheduler
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Initialize the training history dictionary
    history = {
        "max_val_accuracy": 0,
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
    }

    # Training loop
    for epoch in range(epochs):
        model.train()

        total_train_loss = 0
        train_correct = 0

        total_val_loss = 0
        val_correct = 0

        # Iterate over the training dataset
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_correct += (torch.argmax(pred, 1) == y).sum().item()

        with torch.no_grad():
            model.eval()
            # Iterate over the validation dataset
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                total_val_loss += loss.item()
                val_correct += (torch.argmax(pred, 1) == y).sum().item()

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Update the training history
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Save the model if the validation accuracy improves
        if history["max_val_accuracy"] < val_accuracy:
            history["max_val_accuracy"] = val_accuracy
            torch.save(model.state_dict(), save_path)

        # Print epoch information
        print(f"[INFO] EPOCH: {epoch + 1}/{epochs}")
        print(f"[INFO] Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}")
        print(f"[INFO] Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}\n")

    # Plotting the graphs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Saving the image
    image_path = save_path.replace(".pth", ".png")
    plt.savefig(image_path)

    return history, image_path
