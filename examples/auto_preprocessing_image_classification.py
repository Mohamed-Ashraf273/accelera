import json
from pathlib import Path

import pandas as pd
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

EXAMPLES_DIR = Path(__file__).resolve().parent
class ClassificationTraining:
    def __init__(self, dataset_name, folder_path, num_classes):
        self.logs = [dataset_name]
        self.folder_path = folder_path
        self.num_classes = num_classes

    def load_model_resenet(self):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        model.load_state_dict(
            torch.load(f"{self.folder_path}/best_model.pth", map_location="cpu")
        )
        return model

    def inference(self, images, image_class_names):
        model = self.load_model_resenet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        testing_loader, _ = ClassificationImageTestingPreprocessing(
            images, folder_path=self.folder_path, image_class_names=image_class_names
        ).common_preprocessing()
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            print("predicted", predicted)
            print("corrected", labels)

    def train(self, model, train_loader, val_loader, epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.to(device)
        best_accuracy = 0.0
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
            if val_accurcy > best_accuracy:
                best_accuracy = val_accurcy
                torch.save(model.state_dict(), f"{self.folder_path}/best_model.pth")

            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accurcy:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accurcy:.4f}"
            )
            self.logs.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_accurcy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accurcy,
                }
            )

    def pretrained_model(self):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def handle_data(self, train_folder, val_folder, augment, image_size):
        preprocessor = ClassificationImageTrainingPreprocessing(
            training_folder_images=train_folder,
            validation_folder_images=val_folder,
            folder_path=self.folder_path,
            augment=augment,
            split_training=True,
            batch_size=32,
            images_size=image_size,
        )
        train_loader, val_loader = preprocessor.common_preprocessing()
        return train_loader, val_loader


def get_data_set_info():
    with open(EXAMPLES_DIR/"auto_preprocessing_ds.json", "r") as f:
        ds = json.loads(f.read())["image_dataset"]["classification"]
    return ds


def main():
    ds = get_data_set_info()
    for dataset, info in ds.items():
        train_folder = EXAMPLES_DIR/info["train_folder"]
        val_folder = info.get("val_folder", None)
        if val_folder:
            val_folder=EXAMPLES_DIR/val_folder
        
        folder_path = EXAMPLES_DIR/info["report_path"]
        print(folder_path)
        augment = info["augment"] == "True"
        is_train = info["train"] == "True"
        n_class = info.get("n_class", None)
        image_size = (
            info["image_size"]["width"],
            info["image_size"]["height"],
        )
        inferernce = info.get("inferernce", None)
        obj = ClassificationTraining(dataset, folder_path, n_class)
        train_loader, val_loader = obj.handle_data(
            train_folder, val_folder, augment, image_size
        )
        if is_train:
            model = obj.pretrained_model()
            obj.train(model, train_loader, val_loader, epochs=20)
        if inferernce is not None:
            inferernce["images"] = [
                EXAMPLES_DIR / imgage for imgage in inferernce["images"]
            ]
            images = inferernce["images"]
            image_class_names = inferernce["image_class_names"]
            obj.inference(images, image_class_names)
        pd.DataFrame(obj.logs).to_csv(f"{EXAMPLES_DIR}/{folder_path}/logs.csv", index=False)


if __name__ == "__main__":
    main()
