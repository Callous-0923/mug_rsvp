+# MUG-RSVP 多模态 RSVP 识别
+
+本项目实现了一个基于脑电 (EEG) 与眼动 (EM) 信号的多模态 RSVP (Rapid Serial Visual Presentation) 目标识别框架，复现并拓展了原始代码中的数据处理、模型结构与训练流程，方便进一步研究多模态信息融合与不确定性建模。
+
+## 核心特性
+
+- **多模态特征抽取**：包含 EEG Inception 风格骨干网络与轻量级眼动卷积分支，分别将脑电与眼动信号编码为 384/128 维嵌入。【F:models/mug_rsvp_net.py†L16-L31】【F:models/eeg_inception.py†L6-L61】【F:models/eye_branch.py†L6-L32】
+- **门控融合与层次蒸馏**：通过门控机制自适应融合单模态预测，并结合层次化自蒸馏与模态不确定性调制提升三分类与二分类任务性能。【F:models/mug_rsvp_net.py†L21-L40】【F:models/fusion_mug.py†L6-L38】【F:losses/hierarchical_kd.py†L1-L17】【F:engine/train.py†L11-L52】
+- **块留出 + K 折交叉验证流程**：对每位被试的每个任务按数据块留出测试，其余块进行 K 折训练与验证，并在测试阶段导出概率矩阵与评估指标。【F:main.py†L24-L82】【F:engine/test.py†L1-L26】
+
+## 目录结构
+
+```
+configs/        默认训练配置（YAML）
+data/           数据集读取与预处理
+engine/         训练、验证、测试及评估工具
+losses/         自蒸馏等自定义损失函数
+models/         多模态网络与子模块实现
+scripts/        命令行训练、评估、结果导出脚本
+utils/          配置加载、随机数种子等辅助函数
+main.py         默认的跨任务/被试训练入口
+```
+
+## 环境准备
+
+- Python ≥ 3.9（建议使用虚拟环境）
+- PyTorch（支持 CUDA 更佳）
+- 其他依赖：`numpy`、`scipy`、`scikit-learn`、`PyYAML`
+
+使用 `pip` 安装示例：
+
+```bash
+pip install torch torchvision torchaudio
+pip install numpy scipy scikit-learn pyyaml
+```
+
+## 数据准备
+
+项目默认假设数据根目录包含三个任务 `A/B/C`，每个任务下分别存放 EEG (`.npz`) 与 EM (`.mat`) 数据：
+
+```
+<data_root>/
+  ├── A/
+  │   ├── EEG/S1_1.npz, S1_2.npz, ...
+  │   └── EM/S1_1.mat, S1_2.mat, ...
+  ├── B/
+  └── C/
+```
+
+- EEG `.npz` 文件需包含 `data` (N, C_eeg, T) 与 `label` 数组。
+- EM `.mat` 文件需包含 `data` (N, T, C_eye_raw 或 N, C_eye_raw, T) 与 `label`。
+- 数据将被重采样至 `resample_hz`，并在通道维拼接后标准化生成三分类与二分类标签。【F:data/datasets.py†L8-L59】
+
+## 配置文件
+
+默认配置位于 `configs/default.yaml`，涵盖数据路径、任务/被试列表、训练超参及损失系数。【F:configs/default.yaml†L1-L36】
+
+> **重要提示**：`utils/config.load_config()` 默认从绝对路径 `D:/files/codes/mug_rsvp/configs/default.yaml` 读取配置。如需在其他环境运行，请修改该路径或在自定义脚本中显式调用 `load_config(<本地配置路径>)`。【F:utils/config.py†L5-L23】
+
+常见可调参数包括：
+
+- `tasks` / `subjects` / `blocks`：需要参与训练与评估的任务、被试与数据块列表。
+- `epochs`、`batch_size`、`lr`、`patience`：训练轮数、批量大小、学习率与早停容忍度。
+- `lambda_cls_eeg_eye`、`lambda_hsd`、`lambda_gate`：多头分类、层次蒸馏与门控损失权重。
+- `mc_dropout_passes`：MC Dropout 采样次数，用于估计模态不确定性。【F:engine/uncertainty.py†L1-L33】
+
+## 运行训练
+
+1. 准备好配置文件并确保 `save_dir` 可写。
+2. （可选）将配置路径写入 `utils/config.py` 或自行编写脚本传入配置。
+3. 执行主训练脚本：
+
+```bash
+python main.py
+```
+
+脚本会自动完成以下流程：
+
+- 为每个任务、被试与数据块构建数据集与 DataLoader。【F:main.py†L24-L55】
+- 计算类别权重以缓解样本不平衡。【F:main.py†L47-L53】
+- 使用 Stratified K-Fold 进行交叉验证训练，并在验证集上监控平衡准确率调度学习率。【F:main.py†L56-L85】
+- 保存验证损失最优的模型用于测试阶段。【F:main.py†L78-L85】
+
+如需更灵活的命令行流程，可参考 `scripts/train_subject_cv.py` 的 `run_train` 函数，它提供了等效的块留出 + K 折训练逻辑，便于在自定义入口中调用。【F:scripts/train_subject_cv.py†L1-L91】
+
+## 结果导出与评估
+
+- 测试阶段会导出 `*_scores.mat`（包含各模态/融合的概率与标签）与 `*_metrics.json`（包含 Accuracy、Balanced Accuracy、混淆矩阵）。【F:engine/test.py†L1-L26】
+- 使用 `python scripts/evaluate.py` 可汇总所有 JSON 结果生成 `summary.csv` 与 `summary.json`，便于统计多次实验的均值/标准差。【F:scripts/evaluate.py†L1-L39】
+- 使用 `python scripts/export_results.py` 可将 `.mat` 概率文件转换为 CSV 进行进一步分析。【F:scripts/export_results.py†L1-L30】
+
+## 复现建议
+
+- 建议使用 GPU 训练，并适当调整 `batch_size` 与 `mc_dropout_passes` 以平衡速度与性能。
+- 若仅使用单模态，可在 `models/mug_rsvp_net.py` 中改写对应分支或在训练循环中屏蔽相应损失项。
+- 通过调整 `lambda_*` 系数可以控制多头监督、蒸馏与门控的一致性约束力度。【F:engine/train.py†L35-L52】
+
+## 许可与引用
+
+代码仅用于研究用途。如在论文或项目中使用，请引用原始 RSVP 数据集与相关多模态融合工作的成果。
