# D2：筛查真实混凝土裂缝图像

本仓库是 AI 工程营 **Day 2** 的独立作业。目标是帮助设施维护团队制作一个**照片初筛工具**：用程序把混凝土表面照片分成“可能有裂缝 / 没有裂缝”两组，把可能有裂缝的优先交给人工复核，并重点检查**漏检裂缝**。这是一个图像二分类问题。

## 1. 问题与使用者

- **使用者**：需要先筛选大量照片、再安排人工复核的设施维护团队
- **真实输入**：Kaggle Concrete Crack Images（`Positive` / `Negative` 各 20,000 张真实混凝土表面图像）
- **需要的输出**：在同一份固定测试集上比较“多数类基线”与小型 CNN（SmallCNN），并给出被漏检 / 误报的真实图片
- **最重要错误**：假阴性（真实有裂缝但被预测没有）——漏检会让需要人工查看的照片没有被优先发现
- **边界**：输出**只用于安排人工复核**，不能替代现场检查、工程师判断或安全决策

## 2. 真实数据

- **来源**：https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- **位置**：`data/raw/` 下解压后含 `Positive` 与 `Negative` 两个子文件夹的目录（不改文件名）
- **检查命令**：`python train.py --check-data`
- **预期输出**：`REAL DATA CHECK PASSED`、`positive_images: 20000`、`negative_images: 20000`

> ⚠️ 必须使用指定真实来源，不得用生成数据替代；`data/raw/` 已被 `.gitignore` 忽略，不提交原始大数据。

## 3. 环境与安装

- Python 3.13（要求 3.10+）
- 安装依赖：

```powershell
python -m pip install -r requirements.txt
```

依赖：`torch>=2.5`、`torchvision>=0.20`、`matplotlib>=3.9`。

## 4. 运行

在仓库根目录打开终端，依次运行：

```powershell
# 1) 数据检查
python train.py --check-data

# 2) 基线（多数类）、全连接 MLP、候选（SmallCNN），使用同一划分 seed
python train.py --model baseline
python train.py --model mlp --epochs 8
python train.py --model cnn --epochs 2

# 3) 架构对比 + 超参数搜索（多数类 / MLP / SmallCNN / 多个 CNN 配置）
python search.py

# 4) 测试
python -m unittest discover -s tests -v
```

`train.py` 参数：`--data`（数据根目录，默认 `data/raw`）、`--model`（`baseline` / `cnn` / `mlp`）、`--epochs`（训练轮数，默认 2）、`--max-per-class`（每类样本数，默认 600）、`--seed`（划分种子，默认 2026）。

## 5. 预期输出

`python train.py --model ...` 会：

- 打印含指标与错误列表的 JSON；
- 写 `runs/baseline.json` 或 `runs/cnn.json`（指标存档）；
- 写 `runs/baseline-errors.png` 或 `runs/cnn-errors.png`（前 6 张错误图像网格）。

同一划分（`SEED=2026`，每类 600 张，75% 训练 / 25% 测试）上的结果：

| 指标 | 基线 | 候选（SmallCNN） |
| --- | ---: | ---: |
| accuracy | 0.5000 | 0.8133 |
| crack_recall | 1.0000 | 0.7133 |
| crack_precision | 0.5000 | 0.8917 |
| false_negative_cracks（漏检） | 0 | 43 |
| false_positive（误报） | 150 | 13 |
| 混淆矩阵 | [[0,150],[0,150]] | [[137,13],[43,107]] |

（数字来自 `runs/baseline.json` 与 `runs/cnn.json`，与 `report.md` 一致。）

**解读**：基线"永远猜裂缝"从不漏检（recall=1.0），但把 300 张测试图里的 150 张无裂缝图全部误报，复核队列被假警报淹没；CNN 只误报 13 张，但漏检 43 张真裂缝——这正是筛查工具要重点检查的风险。

### 5.1 架构对比与超参数搜索（`python search.py`）

在同一真实数据、同一划分（`SEED=2026`，每类 600，75/25）上，`search.py` 对比了多数类基线、全连接 MLP、起点 SmallCNN 与多个 CNN 配置，结果写入 `runs/search-results.json` 与 `runs/search-accuracy.png`：

| 配置 | accuracy | crack_recall | crack_precision | 漏检 |
| --- | ---: | ---: | ---: | ---: |
| baseline-多数类 | 0.5000 | 1.0000 | 0.5000 | 0 |
| MLP-256/128（全连接，8 轮） | 0.6433 | 0.2867 | 1.0000 | 107 |
| CNN-2conv-8/16（SmallCNN 起点，2 轮） | 0.8133 | 0.7133 | 0.8917 | 43 |
| CNN-3conv-16/32/64 k3（4 轮） | 0.9667 | 0.9467 | 0.9861 | 8 |
| CNN-3conv-32/64/128 k3（4 轮） | 0.9733 | 0.9533 | 0.9931 | 7 |
| CNN-4conv-32/64/128/256 k3（4 轮） | 0.9633 | 0.9267 | 1.0000 | 11 |
| **CNN-3conv-32/64/128 k5（4 轮）** | **0.9833** | **0.9667** | **1.0000** | **5** |

**超参数搜索结论**：把层数从 2 层增加到 3–4 层、卷积核数量从 8/16 增到 16–128、卷积核从 3x3 调到 5x5，accuracy 从起点 SmallCNN 的 **0.8133 提升到 0.9833**，漏检从 43 张降到 5 张——证明超参数搜索带来了明显性能提升。

**架构对比结论**：全连接 MLP 打平像素后丢失了 2D 空间结构，8 轮后 accuracy 仅 0.6433、漏检 107 张；而卷积 CNN 用局部卷积核捕捉边缘/纹理，显著优于 MLP。卷积结构对图像任务至关重要。

## 6. 测试

```powershell
python -m unittest discover -s tests -v
```

预期：`Ran 5 tests ... OK`（SmallCNN / CNN / MLP 输出形状、划分逻辑、漏检计数）。

## 7. 限制与边界

- **准确率会误导**：多数类基线可能准确率不低却认不出裂缝，必须看裂缝召回率和混淆矩阵。
- **数据泄漏风险**：随机拆分高度相似的图像块会让训练和测试看到近重复内容，成绩过于乐观；本任务用固定 seed 划分以便复查。
- **不能当结构安全结论**：筛查器只用于安排人工复核，最终判断由现场人员 / 工程师完成。

## 8. 文件结构

```
README.md            本说明
models.py            SmallCNN / 可配置 CNN / MLP 模型实现
train.py             数据检查、划分、基线、CNN/MLP 训练与评估
search.py            架构对比 + 超参数搜索（写 runs/search-*.json/png）
tests/               单元测试
requirements.txt     依赖
report.md            书面报告
presentation.pptx    3 分钟答辩 PPT
submission.json      提交清单
COURSE-README.md     课程原始 starter 说明（参考）
data/raw/            真实数据（不入库）
runs/                指标与错误图片（可重新生成，不入库）
```
