# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.dataset import DualYOLODataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import DualPoseModel, PoseModel
from ultralytics.utils import DEFAULT_CFG, RANK, colorstr


class PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoint shape, OKS sigmas, and keypoint names on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: torch.nn.Module | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (torch.nn.Module, optional): Pretrained model whose weights are loaded into the new model.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = self.set_model_names_for_load(
            PoseModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoint shape, OKS sigmas, and keypoint names on the PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]
        self.model.kpt_oks_sigmas = self.data.get("kpt_oks_sigmas")
        kpt_names = self.data.get("kpt_names")
        if not kpt_names:
            names = list(map(str, range(self.model.kpt_shape[0])))
            kpt_names = {i: names for i in range(self.model.nc)}
        self.model.kpt_names = kpt_names

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose")
        return data


class DualPoseTrainer(PoseTrainer):
    """双属性 pose 训练器(本项目定制): DualYOLODataset 解析双列标签(颜色 车辆类别 ...),
    模型 nc = 车辆类别数(颜色数由 DualPoseHead 的 nc_color 决定),
    损失用 DualPoseLoss(combined = 颜色idx * nc_vehicle + 车辆idx 在损失层拆分监督)。

    用法(不走 YOLO 高层 API, 直接实例化 trainer):
        from ultralytics.models.yolo.pose.train import DualPoseTrainer
        DualPoseTrainer(overrides=dict(model="armor-dual-pose.yaml",
                                       data="dataset_dual.yaml", epochs=150, imgsz=640)).train()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建 DualPoseModel, nc 传车辆类别数(data.yaml 的 nc_vehicle)。"""
        model = self.set_model_names_for_load(
            DualPoseModel(
                cfg,
                nc=self.data.get("nc_vehicle", 7),
                ch=self.data.get("channels"),
                data_kpt_shape=self.data["kpt_shape"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        """模型 names 换成车辆类别名(数据里的 combined 名仅供数据管线内部使用)。"""
        super().set_model_attributes()
        nc_vehicle = self.data.get("nc_vehicle", 7)
        self.model.nc = nc_vehicle
        self.model.names = self.data.get("vehicle_names") or {i: str(i) for i in range(nc_vehicle)}

    def build_dataset(self, img_path, mode="train", batch=None):
        """用 DualYOLODataset 构建数据集(参数与 build_yolo_dataset 的 YOLODataset 分支一致)。"""
        return DualYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """返回支持 combined->车辆类别 折算的 DualPoseValidator。"""
        return yolo.pose.DualPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
