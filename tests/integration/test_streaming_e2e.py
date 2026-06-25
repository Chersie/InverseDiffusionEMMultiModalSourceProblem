"""End-to-end streaming-data test: write shards, fit IncrementalPCA, train one epoch."""

from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import DataLoader

from mpinv.callbacks.logging_cb import LoggingCallback
from mpinv.data.memmap_dataset import (
    MemmapDataset,
    MemmapShard,
    list_shards,
    shard_token,
    write_shard,
)
from mpinv.features.normalisers import StandardScaler
from mpinv.features.pca import IncrementalPCAStream
from mpinv.losses.coef_mse import CoefMSE
from mpinv.models.mlp import MLP, MLPConfig
from mpinv.training.optim import OptimiserConfig, build_optimiser
from mpinv.training.trainer import Trainer, TrainerConfig


@pytest.mark.integration
def test_streaming_e2e(tmp_path, tiny_generator):
    rng = np.random.default_rng(0)

    # 1. write 3 shards of 64 samples each
    tok = shard_token()
    shards: list[MemmapShard] = []
    for i in range(3):
        P, packed = tiny_generator.generate_batch(64, rng)
        shards.append(write_shard(tmp_path, P=P, packed=packed, shard_idx=i, token=tok))
    assert len(list_shards(tmp_path)) == 3

    # 2. Incremental PCA fit on the streamed flattened P
    chunks = []
    for s in shards:
        P_mm = s.open_P()
        chunks.append(np.array(P_mm.reshape(P_mm.shape[0], -1)))
    ipca = IncrementalPCAStream(n_components=4, batch_size=32).fit_chunks(chunks)
    Z = np.concatenate([ipca.transform(c) for c in chunks], axis=0)
    scaler = StandardScaler().fit(Z)
    Z = scaler.transform(Z)

    ds = MemmapDataset(shards=shards, features=Z)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    K = tiny_generator.n_modes
    model = MLP(MLPConfig(input_dim=4, output_dim=4 * K, hidden_size=16, n_hidden_layers=1))
    loss_fn = CoefMSE()
    opt = build_optimiser(model, OptimiserConfig(name="adamw", lr=1e-3))

    trainer = Trainer(TrainerConfig(max_epochs=1, log_every_n_steps=2))
    ctx = trainer.fit(
        model=model,
        train_loader=loader,
        loss_fn=loss_fn,
        optimiser=opt,
        loss_kind="coef",
        callbacks=[LoggingCallback(log_every_n_steps=2)],
    )
    assert np.isfinite(ctx.last_loss)
    assert ctx.global_step > 0


def test_shard_token_unique():
    a = shard_token()
    import time

    time.sleep(0.005)
    b = shard_token()
    assert a != b


def test_dataset_length_matches_shard_lengths(tmp_path, tiny_generator):
    rng = np.random.default_rng(0)
    P1, p1 = tiny_generator.generate_batch(8, rng)
    P2, p2 = tiny_generator.generate_batch(5, rng)
    s1 = write_shard(tmp_path, P=P1, packed=p1, shard_idx=0, token="t")
    s2 = write_shard(tmp_path, P=P2, packed=p2, shard_idx=1, token="t")
    ds = MemmapDataset(shards=[s1, s2])
    assert len(ds) == 13
    x, pk, P = ds[12]
    assert x.shape[0] == P.numel()
