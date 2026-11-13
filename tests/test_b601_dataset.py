import json

import numpy as np
import pytest
import torch
import zarr

from real_robot.dataset import B601ImageDataset, make_split, write_split


@pytest.fixture
def b601_dataset(tmp_path):
    n, length = 5, 24
    replay = zarr.open_group(str(tmp_path / "replay.zarr"), mode="w")
    data = replay.create_group("data")
    values = np.arange(n*length*7, dtype=np.float32).reshape(n*length, 7)/100
    # Episode 0 is excluded and episode 1 is the deterministic held-out episode.
    values[:length] = -1000
    values[length:2*length] = 1000
    data.array("state", values)
    data.array("action", values+1)
    pixels = np.zeros((n*length, 24, 32, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(n*length)[:, None, None]
    pixels[..., 2] = 255
    data.array("img", pixels)
    replay.create_group("meta").array("episode_ends", np.arange(1, n+1)*length)
    (tmp_path / "episodes.json").write_text(json.dumps([{"episode": i} for i in range(n)]))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"candidate_episode_indices": [1, 2, 3, 4]}))
    split_path = tmp_path / "split.json"
    write_split(selection, tmp_path / "episodes.json", split_path, seed=42, n_validation=1)
    return B601ImageDataset(str(tmp_path / "replay.zarr"), str(selection), str(split_path))


def test_fixed_split_has_no_episode_overlap():
    split = make_split(list(range(6, 40)), n_validation=4, seed=42)
    assert split == make_split(list(range(6, 40)), n_validation=4, seed=42)
    assert len(split["train_episode_indices"]) == 30
    assert len(split["validation_episode_indices"]) == 4
    assert not set(split["train_episode_indices"]) & set(split["validation_episode_indices"])


def test_real_joint_dimensions_alignment_and_boundary_padding(b601_dataset):
    dataset = b601_dataset
    sample = dataset[0]
    start = dataset.sampler.indices[0, 0]
    assert sample["obs"]["image"].shape == (2, 3, 24, 32)
    assert sample["obs"]["joint_pos"].shape == (2, 7)
    assert sample["action"].shape == (16, 7)
    assert sample["obs"]["image"].dtype == torch.float32
    np.testing.assert_allclose(sample["obs"]["image"][:, 0, 0, 0].numpy(), start/255)
    np.testing.assert_array_equal(sample["obs"]["image"][:, 2, 0, 0].numpy(), 1)
    np.testing.assert_array_equal(sample["action"][0], dataset.replay_buffer["action"][start])
    torch.testing.assert_close(sample["action"][0], sample["action"][1])
    last = dataset[len(dataset)-1]
    torch.testing.assert_close(last["action"][-7:], last["action"][-1:].expand(7, -1))
    for view in (dataset, dataset.get_validation_dataset()):
        indices = view.sampler.indices
        first = np.searchsorted(dataset.episode_ends, indices[:, 0], side="right")
        final = np.searchsorted(dataset.episode_ends, indices[:, 1]-1, side="right")
        np.testing.assert_array_equal(first, final)
        assert view.episode_mask[first].all()
    assert not dataset.episode_mask[0]


def test_normalizer_uses_only_training_episodes(b601_dataset):
    dataset = b601_dataset
    normalizer = dataset.get_normalizer()
    stats = normalizer.get_input_stats()
    expected = dataset.training_values("state")
    np.testing.assert_allclose(stats["joint_pos"]["min"].numpy(), expected.min(0))
    np.testing.assert_allclose(stats["joint_pos"]["max"].numpy(), expected.max(0))
    assert stats["joint_pos"]["max"].max().item() < 100
    assert stats["joint_pos"]["min"].min().item() > 0
    np.testing.assert_allclose(dataset.get_validation_dataset().get_normalizer()["action"].params_dict["scale"].numpy(),
                               normalizer["action"].params_dict["scale"].numpy())


def test_changed_selection_cannot_silently_reuse_split(b601_dataset):
    dataset = b601_dataset
    with open(dataset.selection_path, "w") as handle:
        json.dump({"candidate_episode_indices": [0, 1, 2, 3, 4]}, handle)
    with pytest.raises(ValueError, match="selection changed"):
        B601ImageDataset(str(dataset.replay_buffer.root.store.path), dataset.selection_path, dataset.split_path)
