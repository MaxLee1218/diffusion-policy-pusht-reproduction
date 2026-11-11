from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import zarr


@pytest.fixture(scope="session")
def smoke_config():
    configs = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(config_dir=str(configs), version_base=None):
        return hydra.compose(config_name="smoke_test")


@pytest.fixture(scope="session")
def dataset(smoke_config, tmp_path_factory):
    rng = np.random.default_rng(42)
    path = tmp_path_factory.mktemp("push-t") / "replay.zarr"
    replay = zarr.open_group(str(path), mode="w")
    data = replay.create_group("data")
    steps = 4 * 24
    data.array("img", rng.integers(0, 256, (steps, 96, 96, 3), dtype=np.uint8))
    data.array("state", rng.uniform(0, 512, (steps, 5)).astype(np.float32))
    data.array("action", rng.uniform(0, 512, (steps, 2)).astype(np.float32))
    replay.create_group("meta").array("episode_ends", [24, 48, 72, 96])
    return hydra.utils.instantiate(smoke_config.task.dataset, zarr_path=str(path))


@pytest.fixture
def policy(smoke_config, dataset):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model = hydra.utils.instantiate(smoke_config.policy)
    model.set_normalizer(dataset.get_normalizer())
    return model


@pytest.fixture
def batch(dataset):
    return next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
