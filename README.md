# PJM Day-Ahead Forecasting Foundation V1

该项目提供一个可复现、可扩展的日前电价预测底座，目标是在 `epftoolbox` 的 `PJM/COMED` 数据上统一完成：

- 数据下载与标准化
- 特征工程与无泄露切分
- `NBEATSx` 调参和滚动回测
- `Seasonal Naive / LEAR / DNN` 基线结果对齐
- 指标、统计检验与报告素材导出

## Environment

建议使用 `Python 3.12`：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev,ml]
```

如果只想运行数据、特征与测试：

```powershell
python -m pip install -e .[dev]
```

`epftoolbox` 不在 PyPI 上，因此不包含在默认 extras 里。需要运行 `LEAR/DNN` 基线时，单独从源码安装：

```powershell
python -m pip install git+https://github.com/jeslago/epftoolbox.git
```

## Pipeline

```powershell
python scripts\prepare_data.py --config configs\pjm_day_ahead_v1.yaml
python scripts\tune_nbeatsx.py --config configs\pjm_day_ahead_v1.yaml
python scripts\backtest_all_models.py --config configs\pjm_day_ahead_v1.yaml
python scripts\evaluate_and_plot.py --config configs\pjm_day_ahead_v1.yaml
python scripts\export_report_assets.py --config configs\pjm_day_ahead_v1.yaml
```

## Notes

- 第一版保持 `epftoolbox` 时间协议，不做 UTC 重映射。
- `LEAR` 与 `DNN` 适配器需要额外安装 `epftoolbox`，且 `DNN` 需要其超参数优化产物。
- 默认不保存每个周重训窗口的模型 checkpoint，只保存最佳超参数、seed 和预测结果。
