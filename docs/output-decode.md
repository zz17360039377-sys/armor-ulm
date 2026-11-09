# 输出解码

输出 `(B, 4 + nc + nkpt*3 + nc_color, A)`，本模型为 **(B, 26, A)**，`A = 8400`(640) / `3549`(416)。

| 通道 | 含义 | 说明 |
|---|---|---|
| `0:4` | cx cy w h | 已解码，**letterbox 后输入图**像素，不要再做 DFL |
| `4:11` | 7 车辆类概率 | 已 sigmoid |
| `11:23` | 4 × (x, y, v) | 已解码到输入像素，`v` 已 sigmoid |
| `23:26` | 3 颜色 | **`.pt` = sigmoid 值；`.onnx` = 原始 logits** |

顺序：车辆 `0 sentry, 1 no1, 2 no2, 3 no3, 4 no4, 5 outpost, 6 base`；颜色 **`0 红, 1 蓝, 2 灰`**。

## 解码

```python
pred = out[0].T                                # (A, 26)
box, cls, kpt, color = pred[:,:4], pred[:,4:11], pred[:,11:23].reshape(-1,4,3), pred[:,23:26]
score, cls_id = cls.max(1), cls.argmax(1)

xyxy = np.stack([box[:,0]-box[:,2]/2, box[:,1]-box[:,3]/2,
                 box[:,0]+box[:,2]/2, box[:,1]+box[:,3]/2], 1)
xyxy[:,0::2] = (xyxy[:,0::2] - dw) / r          # dw/dh 用浮点值, 不要取整
xyxy[:,1::2] = (xyxy[:,1::2] - dh) / r
kpt[...,0] = (kpt[...,0] - dw) / r
kpt[...,1] = (kpt[...,1] - dh) / r
```

letterbox：等比缩放到 `imgsz`，灰边(114) 填充，`r = min(imgsz/h, imgsz/w)`，
`dw/dh` 为浮点 padding。

## 后处理建议

- **逐类 NMS**（IoU 0.5）：相邻装甲板允许重叠，同类才抑制。
- 近距离交叠时补一条**包含抑制**：`IoMin = inter/min(area) ≥ 0.6` 丢小框/低分框。
- `conf` 0.25~0.4 起调；`max_det` 20 足够。
- 颜色置信度偏低时建议回退红蓝二选一 + 时序滤波（灰甲误判代价最小）。
- 只有 `v ≥ 0.5` 的角点参与 PnP 更稳。

## TensorRT

- 仓库 ONNX 为动态 H/W（fp32, opset 17）；固定到 640/416 通常更快，改形状需重建 engine。
- 同架构不同权重（640 vs 416）的 ONNX 图哈希可能相同，**engine 缓存必须按模型名隔离**。
