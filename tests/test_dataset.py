import numpy as np
import torch


def test_windows_align_observations_and_actions_with_replay(dataset, smoke_config):
    horizon = smoke_config.horizon
    assert horizon == 16
    assert smoke_config.n_obs_steps == 2
    assert smoke_config.n_action_steps == 8
    assert (dataset.pad_before, dataset.pad_after) == (1, 7)

    replay = dataset.replay_buffer
    ends = replay.episode_ends[:]
    starts = np.concatenate(([0], ends[:-1]))
    offset = 0
    for episode in np.flatnonzero(dataset.train_mask):
        start, end = starts[episode], ends[episode]
        count = end - start - horizon + 9
        assert count >= 3
        windows = [
            (offset, np.r_[start, start : start + 15]),
            (offset + 1, np.arange(start, start + 16)),
            (offset + count - 1, np.r_[end - 9 : end, np.repeat(end - 1, 7)]),
        ]
        for index, frames in windows:
            sample = dataset[index]
            images = sample["obs"]["image"][:2]
            assert images.shape == (2, 3, 96, 96)
            expected_images = np.moveaxis(replay["img"][frames[:2]], -1, 1) / 255
            np.testing.assert_array_equal(images.numpy(), expected_images)
            np.testing.assert_array_equal(
                sample["obs"]["agent_pos"][:2].numpy(), replay["state"][frames[:2], :2].astype(np.float32)
            )
            assert sample["action"].shape == (16, 2)
            assert sample["action"].dtype == torch.float32
            np.testing.assert_array_equal(sample["action"].numpy(), replay["action"][frames].astype(np.float32))
        offset += count

    assert offset == len(dataset)
    assert len(dataset.get_validation_dataset()) > 0
