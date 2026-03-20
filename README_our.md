# EvLight 项目训练与测试流程（简版）

## 1. 项目入口与核心流程
- 统一入口：`python egllie/main.py --yaml_file=... --log_dir=...`
- 配置加载后会根据 `IS_VIDEO` 自动选择：
  - `IS_VIDEO: false` -> `egllie/core/launch.py`（图像流程）
  - `IS_VIDEO: true` -> `egllie/core/launch_vid.py`（视频流程）
- 数据集由 `DATASET.NAME` 决定（SDE / SDSD，图像 / 视频）。

## 2. 环境准备
仓库中的依赖文件名是带前导空格的 ` requirement.txt`，安装命令如下：

```bash
pip install -r " requirement.txt"
```

## 3. 数据与配置准备
在 `options/train/*.yaml` 或 `options/test/*.yaml` 中重点修改：
- `DATASET.root`: 数据集根目录（代码会自动拼接 `/train` 和 `/test`）
- `RESUME.PATH`:
  - 训练从头开始可留空
  - 测试必须填写模型路径
  - 视频训练通常用图像模型做初始化（yaml 里已有提示）

常见配置对应关系：
- `sde_in / sde_out`：SDE 数据集（室内/室外）
- `sdsd_in / sdsd_out`：SDSD 数据集（室内/室外）
- `*_vid`：视频增强版本

## 4. 训练流程
1. 选择训练配置，例如 `options/train/sde_in.yaml`。
2. 修改 `DATASET.root`（必要时改 `RESUME.PATH`）。
3. 执行对应脚本（示例）：

```bash
sh options/train/sde_in.sh
# 或
sh options/train/sde_in_vid.sh
```

训练产物默认保存在对应 `log/train/*/` 下，包含：
- `checkpoint.pth.tar`
- `model_best.pth.tar`
- `checkpoint-xxx.pth.tar`
- `model-xxx.pth.tar`

## 5. 测试流程
1. 选择测试配置，例如 `options/test/sde_in.yaml` 或 `options/test/sde_in_vid.yaml`。
2. 修改：
   - `RESUME.PATH` 为待评估模型
   - `DATASET.root` 为数据集根目录
3. 运行测试。

建议直接用主入口命令（更稳妥）：

```bash
python egllie/main.py \
  --yaml_file="options/test/sde_in.yaml" \
  --log_dir="./log/test/sde_in/" \
  --alsologtostderr=True
```

视频测试示例：

```bash
python egllie/main.py \
  --yaml_file="options/test/sde_in_vid.yaml" \
  --log_dir="./log/test/sde_in_vid/" \
  --alsologtostderr=True
```

说明：
- 部分 `options/test/*.sh`（非 `*_vid`）当前指向 `options/release/*.yaml`，仓库内无该目录，建议直接使用上面的主入口命令，或自行修正脚本路径。
- 测试时若开启 `VISUALIZE`，预测图会输出到 `log_dir/epoch-best/<seq_name>/`。
- 指标（PSNR/SSIM/PSNR_star）会打印在日志中。

