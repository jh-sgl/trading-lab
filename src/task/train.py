import lightning as L

from datamodule import BasicDataModule, ConsecutiveCandleDataset
from model.callback import BacktesterCallback
from model.dlinear import DLinearModel

if __name__ == "__main__":
    debug = False

    input_columns = [
        # "future_price_open",
        # "future_price_close",
        # "future_price_high",
        # "future_price_low",
        "future_price_delta",
    ]
    label_columns = ["future_price_delta"]
    meta_columns = ["time"]

    lookback_window = 100
    consecutive_n = 3
    train_num_workers = 48 if not debug else 0
    val_num_workers = 48 if not debug else 0
    softmax_tau = 0.1

    model = DLinearModel(
        seq_len=lookback_window,
        pred_len=24,
        input_ch=len(input_columns),
        moving_avg_kernel_size=25,
        lr=1e-5,
    )

    train_dataset = ConsecutiveCandleDataset(
        data_fp="/data/jh/AlphaSearch_Naive/data/preprocessed_jh/v2.parquet",
        input_columns=input_columns,
        label_columns=label_columns,
        hold_threshold=0.05,
        softmax_tau=softmax_tau,
        date_range=("2015-01-01", "2020-12-31"),
        resample_rule="5min",
        lookback_window=lookback_window,
        consecutive_n=consecutive_n,
    )

    val_dataset = ConsecutiveCandleDataset(
        data_fp="/data/jh/AlphaSearch_Naive/data/preprocessed_jh/v2.parquet",
        input_columns=input_columns,
        label_columns=label_columns,
        hold_threshold=0.05,
        softmax_tau=softmax_tau,
        date_range=("2021-01-01", "2024-12-31"),
        resample_rule="5min",
        lookback_window=lookback_window,
        consecutive_n=consecutive_n,
    )

    datamodule = BasicDataModule(
        train_dataset=train_dataset,
        train_batch_size=512,
        train_num_workers=train_num_workers,
        val_dataset=val_dataset,
        val_batch_size=512,
        val_num_workers=val_num_workers,
    )

    trainer = L.Trainer(devices=1, accelerator="gpu", callbacks=[BacktesterCallback()], check_val_every_n_epoch=10)
    trainer.fit(model=model, datamodule=datamodule)
