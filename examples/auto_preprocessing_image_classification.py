import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from accelera.src.automl.core.classification_image_testing_preprocessing import (  # noqa: E501
    ClassificationImageTestingPreprocessing,
)
from accelera.src.automl.core.classification_image_training_preprocessing import (  # noqa: E501
    ClassificationImageTrainingPreprocessing,
)


class ClassificationTraining:
    def test(self, model, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        accurcy = len = 0.0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, y_pred = torch.max(outputs.data, 1)
                accurcy += (y_pred == labels).sum().item()
                len += labels.size(0)
        accurcy = accurcy / len
        return accurcy

    def train(self, model, train_loader, val_loader, epochs, logs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.to(device)

        for epoch in range(epochs):
            train_loss = train_accurcy = train_len = 0.0
            val_loss = val_accurcy = val_len = 0.0
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_len += labels.size(0)
                _, y_pred = torch.max(outputs.data, 1)
                train_accurcy += (y_pred == labels).sum().item()
            train_accurcy = train_accurcy / train_len
            train_loss = train_loss / train_len
            with torch.no_grad():
                model.eval()
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_len += labels.size(0)
                    _, y_pred = torch.max(outputs.data, 1)
                    val_accurcy += (y_pred == labels).sum().item()
            val_accurcy = val_accurcy / val_len
            val_loss = val_loss / val_len
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accurcy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accurcy:.4f}"
            )
            logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accurcy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accurcy,
                }
            )
        return model

    def pretrained_model(self, num_classes):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def handle_data(self, train_folder, val_folder, folder_path, augment):
        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=train_folder,
            validation_folder_images=val_folder,
            folder_path=folder_path,
            augment=augment,
            split_training=True,
        )
        train_loader, val_loader = preprocessor.common_preprocessing()
        num_classes = len(preprocessor.label2class_mapping)
        return train_loader, val_loader, num_classes


def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        ds = json.loads(f.read())["image_dataset"]["classification"]
    return ds


def main():
    logs = []
    ds = get_data_set_info()
    for dataset, info in ds.items():
        logs.append({"dataset": dataset, "info": info})
        train_folder = info["train_folder"]
        val_folder = info.get("val_folder", None)
        folder_path = info["report_path"]
        augment = info["augment"] == "True"
        is_train = info["train"] == "True"
        inference=info.get("inference", None)
        obj = ClassificationTraining()
        train_loader, val_loader, num_classes = obj.handle_data(
            train_folder, val_folder, folder_path, augment
        )
        if is_train:
            model = obj.pretrained_model(num_classes)
            trained_model = obj.train(model, train_loader, val_loader, epochs=5, logs=logs)
    print(logs)
if __name__ == "__main__":
    main()