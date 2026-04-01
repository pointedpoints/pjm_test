from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hyperopt.exceptions import AllTrialsFailed

from pjm_forecast.prepared_data import EPF_ALIAS_MAP

from .base import ForecastModel


def _to_epftoolbox_frame(history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
    available = pd.concat([history_df, future_df], axis=0).copy()
    available = available.rename(columns=EPF_ALIAS_MAP)
    available.loc[future_df.index, "Price"] = np.nan
    available = available.set_index("ds")
    return available.loc[:, ["Price", "Exogenous 1", "Exogenous 2"]]


def _dnn_trials_filename(
    experiment_id: int,
    nlayers: int,
    dataset: str,
    years_test: int,
    shuffle_train: bool,
    data_augmentation: bool,
    calibration_window_years: int,
) -> str:
    return (
        f"DNN_hyperparameters_nl{nlayers}"
        f"_dat{dataset}"
        f"_YT{years_test}"
        f"{'_SF' if shuffle_train else ''}"
        f"{'_DA' if data_augmentation else ''}"
        f"_CW{calibration_window_years}_{experiment_id}"
    )


def _patch_epftoolbox_dnn_keras_compat() -> None:
    from epftoolbox.models import _dnn as dnn_module  # type: ignore
    from tensorflow.keras.layers import AlphaDropout, BatchNormalization, Dense, Dropout, Input, LeakyReLU, PReLU
    from tensorflow.keras.models import Model
    from tensorflow.keras import optimizers as keras_optimizers

    if getattr(dnn_module.DNNModel, "_pjm_keras_compat_patched", False):
        return

    def _wrap_optimizer(optimizer_cls):
        def _factory(*args, **kwargs):
            if "lr" in kwargs and "learning_rate" not in kwargs:
                kwargs["learning_rate"] = kwargs.pop("lr")
            return optimizer_cls(*args, **kwargs)

        return _factory

    dnn_module.kr.optimizers.Adam = _wrap_optimizer(keras_optimizers.Adam)
    dnn_module.kr.optimizers.RMSprop = _wrap_optimizer(keras_optimizers.RMSprop)
    dnn_module.kr.optimizers.Adagrad = _wrap_optimizer(keras_optimizers.Adagrad)
    dnn_module.kr.optimizers.Adadelta = _wrap_optimizer(keras_optimizers.Adadelta)

    def _build_model(self):
        input_shape = (self.n_features,)
        past_data = Input(shape=input_shape)
        past_dense = past_data
        if self.activation == "selu":
            self.initializer = "lecun_normal"

        for neurons in self.neurons:
            dense_kwargs = {
                "kernel_initializer": self.initializer,
                "kernel_regularizer": self._reg(self.lambda_reg),
            }
            if self.activation == "LeakyReLU":
                past_dense = Dense(neurons, activation="linear", **dense_kwargs)(past_dense)
                past_dense = LeakyReLU(alpha=0.001)(past_dense)
            elif self.activation == "PReLU":
                past_dense = Dense(neurons, activation="linear", **dense_kwargs)(past_dense)
                past_dense = PReLU()(past_dense)
            else:
                past_dense = Dense(neurons, activation=self.activation, **dense_kwargs)(past_dense)

            if self.batch_normalization:
                past_dense = BatchNormalization()(past_dense)

            if self.dropout > 0:
                if self.activation == "selu":
                    past_dense = AlphaDropout(self.dropout)(past_dense)
                else:
                    past_dense = Dropout(self.dropout)(past_dense)

        output_layer = Dense(
            self.outputShape,
            kernel_initializer=self.initializer,
            kernel_regularizer=self._reg(self.lambda_reg),
        )(past_dense)
        return Model(inputs=[past_data], outputs=[output_layer])

    dnn_module.DNNModel._build_model = _build_model
    dnn_module.DNNModel._pjm_keras_compat_patched = True


def ensure_dnn_hyperparameters(
    experiment_id: int,
    hyperparameter_dir: str,
    dataset_dir: str,
    nlayers: int,
    dataset: str,
    years_test: int,
    shuffle_train: bool,
    data_augmentation: bool,
    calibration_window_years: int,
    hyperopt_max_evals: int,
) -> Path:
    trials_path = Path(hyperparameter_dir) / _dnn_trials_filename(
        experiment_id=experiment_id,
        nlayers=nlayers,
        dataset=dataset,
        years_test=years_test,
        shuffle_train=shuffle_train,
        data_augmentation=data_augmentation,
        calibration_window_years=calibration_window_years,
    )
    if trials_path.exists():
        return trials_path

    try:
        from epftoolbox.models._dnn_hyperopt import hyperparameter_optimizer  # type: ignore
    except ImportError as exc:
        raise ImportError("DNN hyperparameter generation requires epftoolbox hyperopt dependencies.") from exc

    _patch_epftoolbox_dnn_keras_compat()
    hyperparameter_optimizer(
        path_datasets_folder=dataset_dir,
        path_hyperparameters_folder=hyperparameter_dir,
        new_hyperopt=1,
        max_evals=hyperopt_max_evals,
        nlayers=nlayers,
        dataset=dataset,
        years_test=years_test,
        calibration_window=calibration_window_years,
        shuffle_train=int(shuffle_train),
        data_augmentation=int(data_augmentation),
        experiment_id=experiment_id,
    )
    if not trials_path.exists():
        raise FileNotFoundError(f"DNN hyperparameter search did not create expected trials file: {trials_path}")
    return trials_path


def _default_dnn_hyperparameters(n_exogenous_inputs: int, nlayers: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "batch_normalization": False,
        "dropout": 0.1,
        "lr": 0.001,
        "seed": 7,
        "activation": "relu",
        "init": "glorot_uniform",
        "reg": None,
        "lambdal1": 0.0,
        "scaleX": "Invariant",
        "scaleY": "Invariant",
        "In: Day": True,
        "In: Price D-1": True,
        "In: Price D-2": True,
        "In: Price D-3": True,
        "In: Price D-7": True,
    }
    default_neurons = [256, 128, 64, 64, 64]
    for layer in range(1, nlayers + 1):
        params[f"neurons{layer}"] = default_neurons[layer - 1]

    for exog in range(1, n_exogenous_inputs + 1):
        params[f"In: Exog-{exog} D"] = True
        params[f"In: Exog-{exog} D-1"] = True
        params[f"In: Exog-{exog} D-7"] = True
    return params


def _build_fallback_dnn(
    experiment_id: int,
    hyperparameter_dir: str,
    nlayers: int,
    dataset: str,
    years_test: int,
    shuffle_train: bool,
    data_augmentation: bool,
    calibration_window_years: int,
    n_exogenous_inputs: int = 2,
):
    from epftoolbox.models import DNN  # type: ignore

    model = DNN.__new__(DNN)
    model.path_hyperparameter_folder = hyperparameter_dir
    model.experiment_id = experiment_id
    model.nlayers = nlayers
    model.years_test = years_test
    model.shuffle_train = int(shuffle_train)
    model.data_augmentation = int(data_augmentation)
    model.dataset = dataset
    model.calibration_window = calibration_window_years
    model.best_hyperparameters = _default_dnn_hyperparameters(
        n_exogenous_inputs=n_exogenous_inputs,
        nlayers=nlayers,
    )
    model.scaler = None
    model.model = None
    return model


@dataclass
class LEARModel(ForecastModel):
    calibration_window_days: int
    name: str = "lear"

    def __post_init__(self) -> None:
        try:
            from epftoolbox.models import LEAR  # type: ignore
        except ImportError as exc:
            raise ImportError("LEARModel requires epftoolbox to be installed.") from exc
        self._model = LEAR(calibration_window=self.calibration_window_days)

    def _safe_recalibrate_predict(self, available_df: pd.DataFrame, next_day: pd.Timestamp) -> np.ndarray:
        df_train = available_df.loc[: next_day - pd.Timedelta(hours=1)]
        df_train = df_train.iloc[-self.calibration_window_days * 24 :]
        df_test = available_df.loc[next_day - pd.Timedelta(weeks=2) :, :]
        x_train, y_train, x_test = self._model._build_and_split_XYs(
            df_train=df_train,
            df_test=df_test,
            date_test=next_day,
        )

        self._model.recalibrate(Xtrain=x_train, Ytrain=y_train)
        y_pred = np.zeros(24, dtype=float)
        x_no_dummies = self._model.scalerX.transform(x_test[:, :-7])
        x_test = x_test.copy()
        x_test[:, :-7] = x_no_dummies

        for hour in range(24):
            prediction = self._model.models[hour].predict(x_test)
            y_pred[hour] = float(np.asarray(prediction).reshape(-1)[0])

        y_pred = self._model.scalerY.inverse_transform(y_pred.reshape(1, -1))
        return np.asarray(y_pred).reshape(-1)

    def fit(self, train_df: pd.DataFrame) -> None:
        self._latest_train = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        available_df = _to_epftoolbox_frame(history_df, future_df)
        next_day = future_df["ds"].min()
        prediction = self._safe_recalibrate_predict(available_df=available_df, next_day=next_day)
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": prediction})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"calibration_window_days": self.calibration_window_days}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LEARModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class DNNModel(ForecastModel):
    experiment_id: int
    hyperparameter_dir: str
    dataset_dir: str
    nlayers: int
    dataset: str
    years_test: int
    shuffle_train: bool
    data_augmentation: bool
    calibration_window_years: int
    auto_generate_hyperparameters: bool
    hyperopt_max_evals: int
    name: str = "dnn"
    used_fallback_hyperparameters: bool = False

    def __post_init__(self) -> None:
        _patch_epftoolbox_dnn_keras_compat()
        try:
            from epftoolbox.models import DNN  # type: ignore
        except ImportError as exc:
            raise ImportError("DNNModel requires epftoolbox to be installed.") from exc
        try:
            if self.auto_generate_hyperparameters:
                ensure_dnn_hyperparameters(
                    experiment_id=self.experiment_id,
                    hyperparameter_dir=self.hyperparameter_dir,
                    dataset_dir=self.dataset_dir,
                    nlayers=self.nlayers,
                    dataset=self.dataset,
                    years_test=self.years_test,
                    shuffle_train=self.shuffle_train,
                    data_augmentation=self.data_augmentation,
                    calibration_window_years=self.calibration_window_years,
                    hyperopt_max_evals=self.hyperopt_max_evals,
                )
            self._model = DNN(
                experiment_id=self.experiment_id,
                path_hyperparameter_folder=self.hyperparameter_dir,
                nlayers=self.nlayers,
                dataset=self.dataset,
                years_test=self.years_test,
                shuffle_train=int(self.shuffle_train),
                data_augmentation=int(self.data_augmentation),
                calibration_window=self.calibration_window_years,
            )
        except (FileNotFoundError, AllTrialsFailed, ValueError) as exc:
            self.used_fallback_hyperparameters = True
            self._model = _build_fallback_dnn(
                experiment_id=self.experiment_id,
                hyperparameter_dir=self.hyperparameter_dir,
                nlayers=self.nlayers,
                dataset=self.dataset,
                years_test=self.years_test,
                shuffle_train=self.shuffle_train,
                data_augmentation=self.data_augmentation,
                calibration_window_years=self.calibration_window_years,
            )

    def fit(self, train_df: pd.DataFrame) -> None:
        self._latest_train = train_df.copy()

    def predict(self, history_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        available_df = _to_epftoolbox_frame(history_df, future_df)
        next_day = future_df["ds"].min()
        prediction = self._model.recalibrate_and_forecast_next_day(df=available_df, next_day_date=next_day)
        return pd.DataFrame({"ds": future_df["ds"].to_numpy(), "y_pred": np.asarray(prediction).reshape(-1)})

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "hyperparameter_dir": self.hyperparameter_dir,
            "dataset_dir": self.dataset_dir,
            "nlayers": self.nlayers,
            "dataset": self.dataset,
            "years_test": self.years_test,
            "shuffle_train": self.shuffle_train,
            "data_augmentation": self.data_augmentation,
            "calibration_window_years": self.calibration_window_years,
            "auto_generate_hyperparameters": self.auto_generate_hyperparameters,
            "hyperopt_max_evals": self.hyperopt_max_evals,
            "used_fallback_hyperparameters": self.used_fallback_hyperparameters,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DNNModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)
