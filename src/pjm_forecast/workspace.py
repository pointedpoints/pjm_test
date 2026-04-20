from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import optuna
import pandas as pd

from .backtest import get_daily_split_days, run_rolling_backtest
from .config import ProjectConfig, load_config
from .data import prepare_dataset
from .evaluation import compute_metrics
from .evaluation.evaluator import Evaluator
from .model_io import load_model_snapshot, save_model_snapshot_bundle, validate_model_prediction_output
from .models import build_model as build_forecast_model
from .models.base import ForecastModel
from .paths import ensure_project_directories
from .prepared_data import FeatureSchema, PreparedDataset
from .retrieval import RetrievalParams
from .retrieval.runner import RetrievalRunner
SplitName = Literal["validation", "test"]


DEFAULT_MLP_UNIT_SEARCH_OPTIONS = ["256x256", "512x512", "768x768"]


def decode_mlp_units(value):
    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"(\d+)x(\d+)", value)
    if not match:
        raise ValueError(f"Unsupported mlp_units encoding: {value}")

    width_in = int(match.group(1))
    width_out = int(match.group(2))
    return [[width_in, width_out], [width_in, width_out], [width_in, width_out]]


def encode_mlp_units(value) -> str:
    if isinstance(value, str):
        decode_mlp_units(value)
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width_in = int(value[0])
        width_out = int(value[1])
        return f"{width_in}x{width_out}"
    raise ValueError(f"Unsupported mlp_units option: {value!r}")


def resolve_mlp_unit_search_options(tuning_cfg: dict[str, object]) -> list[str]:
    configured = tuning_cfg.get("search_space", {}).get("mlp_units")
    if not configured:
        return list(DEFAULT_MLP_UNIT_SEARCH_OPTIONS)
    return [encode_mlp_units(option) for option in configured]


@dataclass(frozen=True)
class ArtifactStore:
    directories: dict[str, Path]

    def raw_csv(self, filename: str) -> Path:
        return self.directories["raw_data_dir"] / filename

    def feature_store(self) -> Path:
        return self.directories["processed_data_dir"] / "feature_store.parquet"

    def split_boundaries(self) -> Path:
        return self.directories["processed_data_dir"] / "split_boundaries.json"

    def panel(self) -> Path:
        return self.directories["processed_data_dir"] / "panel.parquet"

    def weather_features(self) -> Path:
        return self.directories["processed_data_dir"] / "weather_features.parquet"

    def best_params(self, model_name: str) -> Path:
        return self.directories["hyperparameter_dir"] / f"{model_name}_best_params.json"

    def prediction(self, model_name: str, split: str, seed: int, variant: str | None = None) -> Path:
        suffix = "" if not variant else f"_{variant}"
        return self.directories["prediction_dir"] / f"{model_name}_{split}_seed{seed}{suffix}.parquet"

    def prediction_chunk_dir(self, model_name: str, split: str, seed: int, variant: str | None = None) -> Path:
        return self.directories["prediction_dir"] / "chunks" / self.prediction(model_name, split, seed, variant).stem

    def metrics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_metrics.csv"

    def quantile_diagnostics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_quantile_diagnostics.csv"

    def scenario_diagnostics(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_scenario_diagnostics.csv"

    def dm(self, split: str) -> Path:
        return self.directories["metrics_dir"] / f"{split}_dm.csv"

    def hourly_mae_plot(self, split: str) -> Path:
        return self.directories["plots_dir"] / f"{split}_hourly_mae.png"

    def high_vol_week_plot(self, split: str) -> Path:
        return self.directories["plots_dir"] / f"{split}_high_vol_week.png"

    def retrieval_params(self, model_name: str = "nbeatsx_rag") -> Path:
        return self.directories["hyperparameter_dir"] / f"{model_name}_best_params.json"

    def report_asset(self, name: str) -> Path:
        return self.directories["report_dir"] / name

    def model_snapshot_root(self, name: str = "nbeatsx_snapshot") -> Path:
        return self.directories["artifact_dir"] / "models" / name

    def model_snapshot(self, name: str = "nbeatsx_snapshot") -> Path:
        return self.model_snapshot_root(name)

    def snapshot_manifest(self, name: str = "nbeatsx_snapshot") -> Path:
        return self.model_snapshot_root(name) / "manifest.json"

    def write_metrics(self, split: str, metrics_df: pd.DataFrame) -> Path:
        output_path = self.metrics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)
        return output_path

    def write_quantile_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.quantile_diagnostics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path

    def write_scenario_diagnostics(self, split: str, diagnostics_df: pd.DataFrame) -> Path:
        output_path = self.scenario_diagnostics(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_path, index=False)
        return output_path

    def write_dm(self, split: str, dm_df: pd.DataFrame) -> Path:
        output_path = self.dm(split)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dm_df.to_csv(output_path, index=False)
        return output_path

    def write_plot(self, split: str, kind: str, plot_writer: Callable[[Path], None]) -> Path:
        output_path = self._plot_path(split, kind)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_writer(output_path)
        return output_path

    def write_retrieval_params(
        self,
        model_name: str,
        selected_params: RetrievalParams,
        score_grid: dict[str, float],
        output_model_name: str,
    ) -> Path:
        output_path = self.retrieval_params(model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "selected_params": {
                        "alpha": float(selected_params.alpha),
                        "tau": float(selected_params.tau),
                        "predicted_volatility_threshold": selected_params.predicted_volatility_threshold,
                    },
                    "score_grid": score_grid,
                    "output_model_name": output_model_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path

    def load_retrieval_params(self, model_name: str = "nbeatsx_rag") -> dict[str, object]:
        payload = json.loads(self.retrieval_params(model_name).read_text(encoding="utf-8"))
        required = ["selected_params", "score_grid", "output_model_name"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Retrieval params payload is missing keys: {missing}")
        return payload

    def prediction_runs(self, split: str) -> list["PredictionRun"]:
        runs: list[PredictionRun] = []
        for path in sorted(self.directories["prediction_dir"].glob("*.parquet")):
            metadata = pd.read_parquet(path, columns=["model", "split", "seed"])
            if metadata.empty:
                continue
            run_split = str(metadata["split"].iloc[0])
            if run_split != split:
                continue
            model_name = str(metadata["model"].iloc[0])
            seed = int(metadata["seed"].iloc[0])
            run_name = path.stem
            runs.append(
                PredictionRun(
                    name=run_name,
                    path=path,
                    model=model_name,
                    split=run_split,
                    seed=seed,
                    variant=self._prediction_variant(path.stem, model_name, run_split, seed),
                )
            )
        return runs

    def iter_prediction_files(self, split: str | None = None) -> list[Path]:
        files = sorted(self.directories["prediction_dir"].glob("*.parquet"))
        if split is None:
            return files

        selected: list[Path] = []
        for path in files:
            try:
                frame = pd.read_parquet(path, columns=["split"])
            except (FileNotFoundError, ValueError, KeyError):
                continue
            if frame.empty:
                continue
            if frame["split"].iloc[0] == split:
                selected.append(path)
        return selected

    def export_report_bundle(self, split: str) -> list[Path]:
        report_dir = self.directories["report_dir"]
        report_dir.mkdir(parents=True, exist_ok=True)

        copied: list[Path] = []
        for source in [
            self.metrics(split),
            self.quantile_diagnostics(split),
            self.scenario_diagnostics(split),
            self.dm(split),
            self.hourly_mae_plot(split),
            self.high_vol_week_plot(split),
        ]:
            if source.exists():
                target = self.report_asset(source.name)
                shutil.copy2(source, target)
                copied.append(target)
        return copied

    def export_report_assets(self, split: str) -> list[Path]:
        return self.export_report_bundle(split)

    def _plot_path(self, split: str, kind: str) -> Path:
        if kind == "hourly_mae":
            return self.hourly_mae_plot(split)
        if kind == "high_vol_week":
            return self.high_vol_week_plot(split)
        raise ValueError(f"Unsupported plot kind: {kind!r}")

    def _prediction_variant(self, stem: str, model_name: str, split: str, seed: int) -> str | None:
        canonical_stem = f"{model_name}_{split}_seed{seed}"
        if stem == canonical_stem:
            return None
        if stem.startswith(f"{canonical_stem}_"):
            return stem[len(canonical_stem) + 1 :]
        return stem


@dataclass(frozen=True)
class PredictionRun:
    name: str
    path: Path
    model: str
    split: str
    seed: int
    variant: str | None = None


@dataclass(frozen=True)
class ModelStore:
    artifacts: ArtifactStore

    def snapshot_dir(self, name: str = "nbeatsx_snapshot") -> Path:
        return self.artifacts.model_snapshot(name)

    def _resolve_snapshot_path(self, name_or_path: str | Path) -> Path:
        path = Path(name_or_path)
        if not path.is_absolute() and path.name == str(name_or_path):
            return self.snapshot_dir(str(name_or_path))
        return path

    def save_snapshot(
        self,
        model: ForecastModel,
        *,
        model_name: str,
        name: str = "nbeatsx_snapshot",
        history_df: pd.DataFrame | None = None,
        prediction_horizon: int | None = None,
        prediction_freq: str | None = None,
    ) -> Path:
        output_dir = self.snapshot_dir(name)
        return save_model_snapshot_bundle(
            model,
            model_name=model_name,
            snapshot_path=output_dir,
            history_df=history_df,
            prediction_horizon=prediction_horizon,
            prediction_freq=prediction_freq,
        )

    def load_snapshot(self, name_or_path: str | Path = "nbeatsx_snapshot") -> ForecastModel:
        return load_model_snapshot(self._resolve_snapshot_path(name_or_path))

    def load_nbeatsx_snapshot(self, name_or_path: str | Path = "nbeatsx_snapshot") -> ForecastModel:
        return self.load_snapshot(name_or_path)

    def predict_snapshot(
        self,
        name_or_path: str | Path,
        *,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        model = self.load_snapshot(name_or_path)
        return model.predict(history_df=history_df, future_df=future_df)

    def predict_snapshot_to_parquet(
        self,
        name_or_path: str | Path,
        *,
        history_df: pd.DataFrame,
        future_df: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        model = self.load_snapshot(name_or_path)
        predictions = model.predict(history_df=history_df, future_df=future_df)
        model_name = getattr(model, "name", "snapshot_model")
        validated = validate_model_prediction_output(predictions, future_df=future_df, model_name=str(model_name))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validated.to_parquet(output_path, index=False)
        return output_path


@dataclass
class Workspace:
    config: ProjectConfig
    directories: dict[str, Path]
    artifacts: ArtifactStore
    models: ModelStore
    _loaded_best_params: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def open(cls, config_path: str | Path) -> "Workspace":
        config = load_config(config_path)
        directories = ensure_project_directories(config)
        artifacts = ArtifactStore(directories)
        model_store = ModelStore(artifacts)
        return cls(config=config, directories=directories, artifacts=artifacts, models=model_store)

    def feature_frame(self) -> pd.DataFrame:
        return self.prepared_dataset().feature_df

    def split_boundaries(self) -> dict[str, pd.Timestamp]:
        return self.prepared_dataset().split_boundaries

    def schema(self) -> FeatureSchema:
        return FeatureSchema(self.config)

    def prepared_dataset(self) -> PreparedDataset:
        return PreparedDataset.from_artifacts(
            self.config,
            panel_path=self.artifacts.panel(),
            feature_path=self.artifacts.feature_store(),
            split_boundaries_path=self.artifacts.split_boundaries(),
        )

    def split_days(self, split: SplitName) -> list[pd.Timestamp]:
        return self.prepared_dataset().split_days(split)

    def _raw_build_model(self, model_name: str, seed: int | None = None, disable_ensemble: bool = False):
        return build_forecast_model(
            self.config,
            model_name,
            seed=seed,
            hyperparameter_dir=self.directories["hyperparameter_dir"],
            disable_ensemble=disable_ensemble,
        )

    def _apply_best_params(self, model_name: str = "nbeatsx") -> None:
        best_params_path = self.artifacts.best_params(model_name)
        if model_name in self._loaded_best_params or not best_params_path.exists():
            return

        best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        best_params["mlp_units"] = decode_mlp_units(best_params.get("mlp_units"))
        self.config.models[model_name].update(best_params)
        self._loaded_best_params.add(model_name)

    def build_model(self, model_name: str, *, seed: int | None = None, disable_ensemble: bool = False):
        if model_name == "nbeatsx":
            self._apply_best_params(model_name)
        return self._raw_build_model(model_name, seed=seed, disable_ensemble=disable_ensemble)

    def prepare(self) -> None:
        result = prepare_dataset(self.config, self.directories["raw_data_dir"])
        prepared = result.prepared
        if result.weather_df is not None:
            result.weather_df.to_parquet(self.artifacts.weather_features(), index=False)
        prepared.save(
            panel_path=self.artifacts.panel(),
            feature_path=self.artifacts.feature_store(),
            split_boundaries_path=self.artifacts.split_boundaries(),
        )

    def tune_nbeatsx(self) -> None:
        feature_df = self.feature_frame()
        self.schema().validate_nbeatsx_feature_frame(feature_df)
        split_boundaries = self.split_boundaries()
        validation_days = get_daily_split_days(feature_df, split_boundaries, split_name="validation")
        tuning_cfg = self.config.tuning
        use_ensemble_in_tuning = bool(tuning_cfg.get("use_ensemble_in_tuning", False))
        metric_name = str(tuning_cfg.get("metric", "mae")).lower()

        def objective(trial: optuna.Trial) -> float:
            self.config.models["nbeatsx"]["input_size"] = trial.suggest_categorical(
                "input_size",
                tuning_cfg["search_space"]["input_size"],
            )
            self.config.models["nbeatsx"]["learning_rate"] = trial.suggest_float(
                "learning_rate",
                tuning_cfg["search_space"]["learning_rate"][0],
                tuning_cfg["search_space"]["learning_rate"][1],
                log=True,
            )
            self.config.models["nbeatsx"]["batch_size"] = trial.suggest_categorical(
                "batch_size",
                tuning_cfg["search_space"]["batch_size"],
            )
            self.config.models["nbeatsx"]["max_steps"] = trial.suggest_int(
                "max_steps",
                tuning_cfg["search_space"]["max_steps"][0],
                tuning_cfg["search_space"]["max_steps"][1],
            )
            self.config.models["nbeatsx"]["dropout_prob_theta"] = trial.suggest_float(
                "dropout_prob_theta",
                tuning_cfg["search_space"]["dropout"][0],
                tuning_cfg["search_space"]["dropout"][1],
            )
            mlp_units_key = trial.suggest_categorical("mlp_units", resolve_mlp_unit_search_options(tuning_cfg))
            self.config.models["nbeatsx"]["mlp_units"] = decode_mlp_units(mlp_units_key)

            predictions = run_rolling_backtest(
                config=self.config,
                feature_df=feature_df,
                split_name="validation",
                forecast_days=validation_days,
                model_builder=lambda: self._raw_build_model(
                    "nbeatsx",
                    seed=self.config.project["benchmark_seed"],
                    disable_ensemble=not use_ensemble_in_tuning,
                ),
                model_name="nbeatsx",
                seed=self.config.project["benchmark_seed"],
            )
            return float(compute_metrics(predictions)[metric_name])

        storage = tuning_cfg.get("optuna_storage")
        study_name = tuning_cfg.get("optuna_study_name", "nbeatsx_tuning")
        if storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tuning_cfg["n_trials"], catch=(RuntimeError, ValueError))

        self.artifacts.best_params("nbeatsx").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        self._loaded_best_params.discard("nbeatsx")

    def backtest(self, split: SplitName = "test") -> None:
        feature_df = self.feature_frame()
        split_boundaries = self.split_boundaries()
        forecast_days = get_daily_split_days(feature_df, split_boundaries, split_name=split)

        for model_name in self.config.backtest["benchmark_models"]:
            if model_name == "nbeatsx":
                self.schema().validate_nbeatsx_feature_frame(feature_df)
            seeds = self.config.project["random_seeds"] if model_name == "nbeatsx" else [self.config.project["benchmark_seed"]]
            for seed in seeds:
                output_path = self.artifacts.prediction(model_name, split, seed)
                if output_path.exists():
                    continue
                run_rolling_backtest(
                    config=self.config,
                    feature_df=feature_df,
                    split_name=split,
                    forecast_days=forecast_days,
                    model_builder=lambda model_name=model_name, seed=seed: self.build_model(model_name, seed=seed),
                    model_name=model_name,
                    seed=seed,
                    output_path=output_path,
                )

    def evaluate(self, split: SplitName = "test") -> None:
        evaluator = Evaluator(schema=self.schema(), artifacts=self.artifacts)
        bundle = evaluator.load_runs(split)
        metrics_df = evaluator.compute_metrics(bundle)
        evaluator.compute_quantile_diagnostics(bundle)
        evaluator.compute_scenario_diagnostics(bundle)
        evaluator.compute_dm(bundle)
        evaluator.render_plots(bundle, metrics_df, split)

    def export_report(self, split: SplitName = "test") -> list[Path]:
        return self.artifacts.export_report_bundle(split)

    def export_model_snapshot(self, model_name: str = "nbeatsx", snapshot_name: str | None = None) -> Path:
        window_days = self.config.backtest["rolling_window_days"]
        history_df = self.prepared_dataset().latest_history_window(window_days)
        model = self.build_model(model_name, seed=self.config.project["benchmark_seed"])
        model.fit(history_df)
        return self.models.save_snapshot(
            model,
            model_name=model_name,
            name=snapshot_name or f"{model_name}_snapshot",
            history_df=history_df,
            prediction_horizon=self.config.prediction_horizon,
            prediction_freq=self.config.prediction_freq,
        )

    def export_nbeatsx_snapshot(self, name: str = "nbeatsx_snapshot") -> Path:
        return self.export_model_snapshot(model_name="nbeatsx", snapshot_name=name)

    def predict_model_snapshot(
        self,
        *,
        snapshot_name_or_path: str | Path,
        history_path: str | Path,
        future_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        history_df = pd.read_parquet(Path(history_path))
        future_df = pd.read_parquet(Path(future_path))
        return self.models.predict_snapshot_to_parquet(
            snapshot_name_or_path,
            history_df=history_df,
            future_df=future_df,
            output_path=output_path,
        )

    def predict_nbeatsx_snapshot(
        self,
        *,
        snapshot_name_or_path: str | Path = "nbeatsx_snapshot",
        history_path: str | Path,
        future_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        return self.predict_model_snapshot(
            snapshot_name_or_path=snapshot_name_or_path,
            history_path=history_path,
            future_path=future_path,
            output_path=output_path,
        )

    def _load_or_backtest_prediction(
        self,
        feature_df: pd.DataFrame,
        forecast_days: list[pd.Timestamp],
        split_name: str,
        seed: int,
        model_name: str = "nbeatsx",
        variant: str | None = None,
    ) -> pd.DataFrame:
        output_path = self.artifacts.prediction(model_name, split_name, seed, variant)
        if output_path.exists():
            return pd.read_parquet(output_path)

        predictions = run_rolling_backtest(
            config=self.config,
            feature_df=feature_df,
            split_name=split_name,
            forecast_days=forecast_days,
            model_builder=lambda: self.build_model(model_name, seed=seed),
            model_name=model_name,
            seed=seed,
            output_path=output_path,
        )
        return predictions

    def retrieve_nbeatsx(self, split: SplitName = "test") -> None:
        retrieval = self.config.retrieval
        if not retrieval or retrieval.get("enabled") is False:
            return

        base_model_name = self.config.retrieval_base_model_name
        runner = RetrievalRunner(
            self.config,
            self.prepared_dataset(),
            self.artifacts,
            prediction_loader=self._load_or_backtest_prediction,
        )
        runner.tune(base_model=base_model_name, split="validation")
        runner.apply(base_model=base_model_name, split="validation")
        if split == "test":
            runner.apply(base_model=base_model_name, split="test")
