import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from accelera.src.automl.core.segmentation_image_testing_preprocessing import (  # noqa: E501
    SegmentationImageTestingPreprocessing,
)
from accelera.src.automl.core.segmentation_image_training_preprocessing import (  # noqa: E501
    SegmentationImageTrainingPreprocessing,
)


class UnetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UnetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            self.block = UnetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.transpose(x)
            u = torch.cat([x, y], dim=1)
            u = self.block(u)

            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x, y1 = self.enc_block1(x)
        x, y2 = self.enc_block2(x)
        x, y3 = self.enc_block3(x)
        x, y4 = self.enc_block4(x)

        x = self.bottleneck(x)
        x = self.dec_block1(x, y4)
        x = self.dec_block2(x, y3)
        x = self.dec_block3(x, y2)
        x = self.dec_block4(x, y1)

        return self.out(x)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


def dice_score(logits, targets, eps=1e-4):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)

    return dice.mean().item()


def validate(model, val_dataloader, loss_1, loss_2, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in val_dataloader:
            imgs = imgs.to(device)
            masks = masks.float().to(device)

            preds = model(imgs)
            loss = loss_1(preds, masks) + loss_2(preds, masks)

            val_loss += loss.item()
            val_dice += dice_score(preds, masks)
            count += 1

    return val_loss / count, val_dice / count


class SegmentationTraining:
    def __init__(
        self,
        dataset_name,
        folder_path,
    ):
        self.logs = [dataset_name]
        self.folder_path = folder_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        chekpoint = torch.load(
            f"{self.folder_path}/best_model.tar", map_location=self.device
        )
        model = UnetModel()
        model.load_state_dict(chekpoint["model_state"])
        model.to(self.device)
        model.eval()

        print(
            "Loaded epoch:",
            chekpoint.get("epoch"),
            "best_dice:",
            chekpoint.get("best_dice"),
        )
        return model

    def inference(self, val_dataloader):
        model = self.load_model()
        images, masks = next(iter(val_dataloader))
        model.eval()
        indices = range(len(images))
        with torch.no_grad():
            for idx in indices:
                img, true_mask = images[idx], masks[idx]

                img_batch = img.unsqueeze(0).to(self.device)

                logits = model(img_batch)

                probs = torch.sigmoid(logits)

                pred_mask = (probs > 0.5).float()

                img_np = img.permute(1, 2, 0).cpu().numpy()
                true_np = true_mask.squeeze(0).cpu().numpy()
                pred_np = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

                plt.figure(figsize=(12, 3))

                plt.subplot(1, 5, 1)
                plt.title(f"Image (idx={idx})")
                plt.imshow(img_np)
                plt.axis("off")

                plt.subplot(1, 5, 2)
                plt.title("True mask")
                plt.imshow(true_np, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 5, 3)
                plt.title("Pred mask")
                plt.imshow(pred_np, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 5, 4)
                plt.title(f"Overlay (thr={0.5})")
                plt.imshow(img_np)
                plt.imshow(pred_np, alpha=0.4, cmap="Reds")
                plt.axis("off")

                plt.subplot(1, 5, 5)
                plt.title(f"True Overlay (thr={0.5})")
                plt.imshow(img_np)
                plt.imshow(true_np, alpha=0.4, cmap="Reds")
                plt.axis("off")

                plt.show()

    def train(self, train_loader, val_loader, epochs):
        log_interval = 100
        model = UnetModel()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001, weight_decay=0.0001
        )
        loss_1 = nn.BCEWithLogitsLoss()
        loss_2 = SoftDiceLoss()
        model = model.to(self.device)
        best_dice = -1.0
        for _e in range(epochs):
            model.train()
            loss_mean = 0
            lm_count = 0
            for batch_idx, (imgs, masks) in enumerate(train_loader):
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                masks = masks.float()
                predict = model(imgs)
                loss = loss_1(predict, masks) + loss_2(predict, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lm_count += 1
                loss_mean += loss.item()

                if (batch_idx + 1) % log_interval == 0:
                    print(
                        f"Epoch [{_e + 1}/{epochs}] | "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}] | "
                        f"Train loss: {loss_mean / lm_count:.4f}"
                    )

            val_loss, val_dice = validate(
                model, val_loader, loss_1, loss_2, self.device
            )

            print(
                f"Epoch {_e + 1}: "
                f"val_loss = {val_loss:.4f}, val_dice = {val_dice:.4f}"
            )
            self.logs.append(
                {"epoch": _e + 1, "val_loss": val_loss, "val_dice": val_dice}
            )

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "epoch": _e + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_dice": best_dice,
                    },
                    "best_model.tar",
                )

                print(f"best model saved (dice = {best_dice:.4f})")
                self.logs.append(f"best model saved (dice = {best_dice:.4f})")

    def handle_data(
        self,
        train_folder_images,
        val_folder_images,
        training_folder_masks,
        val_folder_masks,
        augment,
        image_size,
    ):
        train_dataloader, val_dataloader = SegmentationImageTrainingPreprocessing(
            training_folder_images=train_folder_images,
            training_folder_masks=training_folder_masks,
            folder_path=self.folder_path,
            binary_mask_threshold=128,
            validation_folder_images=val_folder_images,
            validation_folder_masks=val_folder_masks,
            split_training=True,
            augment=augment,
            val_size=0.2,
            batch_size=2,
            random_state=23,
            images_size=image_size,
            horizontal_flip=True,
            vertical_flip=True,
            rotation=True,
        ).common_preprocessing()
        return train_dataloader, val_dataloader


def get_data_set_info():
    with open("auto_preproceesing_ds.json", "r") as f:
        ds = json.loads(f.read())["image_dataset"]["segmentation"]
    return ds


def main():
    ds = get_data_set_info()
    for dataset, info in ds.items():
        train_folder_images = info["train_folder_images"]

        train_folder_masks = info["train_folder_masks"]
        val_folder_images = info.get("val_folder_images", None)
        val_folder_masks = info.get("val_folder_masks", None)
        folder_path = info["report_path"]
        augment = info["augment"] == "True"
        is_train = info["train"] == "True"
        inferernce = info.get("inferernce", None)
        obj = SegmentationTraining(dataset, folder_path)
        image_size = (
            info["image_size"]["width"],
            info["image_size"]["height"],
        )
        train_loader, val_loader = obj.handle_data(
            train_folder_images,
            val_folder_images,
            train_folder_masks,
            val_folder_masks,
            augment,
            image_size,
        )
        if is_train:
            obj.train(train_loader, val_loader, epochs=20)
            obj.inference(val_loader)
        if inferernce is not None:
            testing_loader, invalid_images = SegmentationImageTestingPreprocessing(
                inferernce["images"], inferernce["masks", folder_path]
            ).common_preprocessing()
            obj.inference(testing_loader)
        pd.DataFrame(obj.logs).to_csv(f"{folder_path}/logs.csv", index=False)


if __name__ == "__main__":
    main()
