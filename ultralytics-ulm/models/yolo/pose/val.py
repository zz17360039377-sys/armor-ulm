# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized
    metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation: dataset `kpt_oks_sigmas` if provided, else OKS_SIGMA for
            COCO keypoints or ones divided by number of keypoints.
        kpt_shape (list[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        postprocess: Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.
        _prepare_batch: Prepare a batch for processing by scaling normalized keypoints to model input image dimensions.
        _process_batch: Return correct prediction matrices from box IoU and keypoint OKS between detections and ground
            truth.
        gather_stats: Gather stats from all GPUs.
        scale_preds: Scale predictions to the original image size.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate pose estimation model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo26n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()

    Notes:
        This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
        for OKS calculation and sets up PoseMetrics for evaluation.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (dict, optional): Dictionary of callback functions to be executed during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.

        Raises:
            ValueError: If dataset `kpt_oks_sigmas` does not contain one positive value per keypoint.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        if sigmas := self.data.get("kpt_oks_sigmas"):  # optional custom OKS sigmas from the dataset YAML
            self.sigma = np.array(sigmas, dtype=np.float32).flatten()
            if len(self.sigma) != nkpt or not np.all(self.sigma > 0):
                raise ValueError(f"'kpt_oks_sigmas' must be {nkpt} positive values, got {sigmas}")
        else:
            self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of
        predictions and reshaping them according to the keypoint shape configuration. The keypoints are reshaped from a
        flattened format to the proper dimensional structure (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing bounding boxes, confidence
                scores, class predictions, and keypoint data.

        Returns:
            (list[dict[str, torch.Tensor]]): List of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Notes:
            The keypoints are extracted from the 'extra' field which contains additional task-specific data beyond
            basic detection.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").reshape(-1, *self.kpt_shape)  # reshape: 非连续张量 view 会报错
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch for processing by scaling normalized keypoints to model input image dimensions.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (dict[str, Any]): Prepared batch with keypoints scaled to model input (letterboxed) image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to the model input (letterboxed) image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        pbatch["keypoints"] = kpts
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrices from box IoU and keypoint OKS between detections and ground truth.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels, 'bboxes'
                for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing the box true positives 'tp' and the pose true positives
                'tp_p', each with shape (N, 10) for 10 IoU/OKS thresholds.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        super().gather_stats()  # gather stats from DetectionValidator
        self._gather_image_metrics(self.metrics.pose)

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls', and 'keypoints'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and batch data, converts the bounding boxes from YOLO format to COCO
        format, and appends the results with keypoints to the internal JSON dictionary (self.jdict).

        Args:
            predn (dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls', and 'keypoints'
                tensors.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        super().pred_to_json(predn, pbatch)
        kpts = predn["keypoints"]
        for i, k in enumerate(kpts.flatten(1, 2).tolist()):
            self.jdict[-len(kpts) + i]["keypoints"] = k  # keypoints

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale boxes and keypoints to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "keypoints": ops.scale_coords(
                pbatch["imgsz"],
                predn["keypoints"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate pose estimation model using COCO JSON format."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
        pred_json = self.save_dir / "predictions.json"  # predictions
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "keypoints"], suffix=["Box", "Pose"])

class DualPoseValidator(PoseValidator):
    """双属性 pose 验证器(本项目定制)。

    数据管线的类别是 combined = 颜色idx * nc_vehicle + 车辆idx(见 DualYOLODataset),
    验证时把 GT 类别折回车辆类别(combined % nc_vehicle)与模型的车辆分支输出对齐;
    颜色分支在预测输出最末尾 nc_color 列, 标准 pose 指标计算不使用它。
    nc_vehicle 从验证参数(data yaml 的同名字段)读取, 默认 7。
    """

    def postprocess(self, preds):
        """双属性验证的 NMS: 关闭 multi_label。

        multi_label 会让一个框按多个类别复制成多行, 而 pose 的 OKS 匹配矩阵是按
        (GT数 × 预测数) 构建的 —— 行数在 _process_batch 之前无法回溯对齐, 会触发
        "size of tensor a must match tensor b" 错误。每框只保留最佳类别对验证
        指标无实质影响(训练目标本身就是单标签)。
        """
        import ultralytics.utils.nms as nms_mod
        outputs = nms_mod.non_max_suppression(
            preds[0] if isinstance(preds, (list, tuple)) else preds,
            self.args.conf,
            self.args.iou,
            nc=self.nc,
            multi_label=False,          # 关键差异: 单标签, 保证行数与后续 OKS 矩阵对齐
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=False,
        )
        nk = self.kpt_shape[0] * self.kpt_shape[1]   # 关键点通道数(如 4*3=12)
        n_color = self._nc_color()
        processed = []
        for x in outputs:
            # NMS 输出行 = [xyxy(4), conf, cls, extra(nk + ncolor)]
            # _inference 是 cat([box, cls, kpts, color]): extra 前 nk 列是关键点, 末尾 ncolor 列是颜色
            pred = {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]}
            pred["keypoints"] = x[:, 6:6 + nk].reshape(-1, *self.kpt_shape)
            pred["color"] = x[:, 6 + nk:6 + nk + n_color]
            processed.append(pred)
        return processed

    def get_dataloader(self, dataset_path, batch_size):
        """独立运行(yolo val / validator())时的数据加载器。

        训练中的验证直接用 trainer.test_loader(已是 DualYOLODataset);
        独立跑必须同样走双属性解析, 否则 18 列标签会被标准解析器整体判 corrupt。
        """
        from ultralytics.data.build import build_dataloader
        from ultralytics.data.dataset import DualYOLODataset
        dataset = DualYOLODataset(
            img_path=dataset_path,
            imgsz=self.args.imgsz,
            batch_size=batch_size,
            augment=False,               # val 不增强
            hyp=self.args,
            rect=True,                   # 矩形推理
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=colorstr("val: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def _dual_head(self):
        """取 DualPoseHead(有的调用点 self.model 尚未赋值, 取不到返回 None)。"""
        try:
            m = getattr(self.model, "ema", self.model)
            head = m.model[-1]
        except Exception:
            return None
        return head if hasattr(head, "nc_color") else None

    def _nc_color(self):
        """颜色通道数: 校验器 self.data(dict) 的 nc_color 优先, 其次模型头, 缺省 2。

        注意: self.args.data 是 yaml 路径字符串而非 dict; 训练中验证时
        postprocess 调用点 self.model 也可能尚未赋值 —— 所以主读 self.data。
        """
        d = getattr(self, "data", None)
        if isinstance(d, dict):
            try:
                return int(d.get("nc_color", 2) or 2)
            except Exception:
                pass
        head = self._dual_head()
        if head is not None:
            return int(head.nc_color)
        return 2

    def _prepare_batch(self, si, batch):
        """Prepare batch, 折算 GT 类别: combined -> 车辆类别idx。"""
        pbatch = super()._prepare_batch(si, batch)
        nc_vehicle = 7
        d = getattr(self, "data", None)
        if isinstance(d, dict):
            try:
                nc_vehicle = int(d.get("nc_vehicle", 7))
            except Exception:
                nc_vehicle = 7
        if nc_vehicle <= 0:
            head = self._dual_head()
            nc_vehicle = int(head.nc) if head is not None else 7
        pbatch["cls"] = pbatch["cls"] % nc_vehicle
        return pbatch
