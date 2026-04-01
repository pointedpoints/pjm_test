# PJM Day-Ahead Forecasting Foundation V1

该项目提供一个可复现、可扩展的日前电价预测底座，目标是在 `epftoolbox` 的 `PJM/COMED` 数据上统一完成：

- 数据下载与标准化
- 特征工程与无泄露切分
- `NBEATSx` 调参和滚动回测
- `Seasonal Naive / LEAR / DNN` 基线结果对齐
- 可选的 retrieval/RAG 实验分支
- 指标、统计检验与报告素材导出

## Environment

建议使用 `uv + Python 3.12`：

```powershell
uv python install 3.12
uv venv --python 3.12
uv sync --extra dev --extra ml
```

如果只想运行数据、特征与测试：

```powershell
uv sync --extra dev
```

`epftoolbox` 不在 PyPI 上，因此不包含在默认 extras 里。需要运行 `LEAR/DNN` 基线时，单独从源码安装：

```powershell
uv pip install git+https://github.com/jeslago/epftoolbox.git
```

## Pipeline

```powershell
uv run python scripts\run_pipeline.py --config configs\pjm_day_ahead_v1.yaml --split test
```

默认主线固定为：

`prepare -> tune -> backtest -> evaluate -> export`

`retrieve_nbeatsx` 是显式实验步骤，不属于默认 benchmark 流水线。

也可以按阶段单独执行：

```powershell
uv run python scripts\prepare_data.py --config configs\pjm_day_ahead_v1.yaml
uv run python scripts\tune_nbeatsx.py --config configs\pjm_day_ahead_v1.yaml
uv run python scripts\backtest_all_models.py --config configs\pjm_day_ahead_v1.yaml
uv run python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_v1.yaml
uv run python scripts\export_report_assets.py --config configs\pjm_day_ahead_v1.yaml
```

可选实验与运维脚本已下沉：

```powershell
uv run python scripts\experiments\retrieve_nbeatsx.py --config configs\pjm_day_ahead_v1.yaml --split test
uv run python scripts\ops\export_nbeatsx_snapshot.py --config configs\pjm_day_ahead_v1.yaml
```

## Notes

- 第一版保持 `epftoolbox` 时间协议，不做 UTC 重映射。
- `LEAR` 与 `DNN` 适配器需要额外安装 `epftoolbox`，且 `DNN` 需要其超参数优化产物。
- 默认不保存每个周重训窗口的模型 checkpoint，只保存最佳超参数、seed 和预测结果。
- `Workspace + ArtifactStore` 是当前主流程边界；脚本层只保留 CLI shim。
- 运行测试使用 `uv run pytest`。
