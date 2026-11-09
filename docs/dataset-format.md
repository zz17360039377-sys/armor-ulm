# 数据集格式

## 目录

```
my_dataset/
├── data.yaml
├── images/{train,val}/*.jpg
└── labels/{train,val}/*.txt     # 与图片同名
```

## 标签（18 列）

```
颜色id 车辆id cx cy w h  lt_x lt_y v  lb_x lb_y v  rb_x rb_y v  rt_x rt_y v
```

- 类别是**两列**，不是组合类别；`cx cy w h` 与角点 `x y` 都归一化到 `[0,1]`。
- 角点顺序 `lt → lb → rb → rt`；`v ∈ {0,1,2}`（不可见/遮挡/可见）。
- 框 = 四角点外接矩形。

示例：

```
0 5 0.500656 0.349694 0.015593 0.008481 0.49335 0.345232 2 0.493089 0.353181 2 0.508384 0.353623 2 0.508471 0.345461 2
```

## data.yaml

```yaml
path: /abs/path/to/my_dataset
train: images/train
val: images/val
nc: 21            # 必须 = nc_color × nc_vehicle
nc_vehicle: 7     # 0哨兵 1~4号 5前哨站 6基地
nc_color: 3       # 0红 1蓝 2灰(死亡)
kpt_shape: [4, 3]
flip_idx: [1, 0, 3, 2]   # 不写会关掉 fliplr/flipud 增强
names: {0: red_sentry, ..., 20: gray_base}   # 21 条，颜色外层、车辆内层
vehicle_names: {0: sentry, ..., 6: base}
```
