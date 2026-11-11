import torch


def test_dataset_produces_padded_vision_and_action_sequences(dataset, smoke_config):
    assert len(dataset) > 0
    assert len(dataset.get_validation_dataset()) > 0
    sample = dataset[0]
    horizon = smoke_config.horizon
    image = sample["obs"]["image"]
    assert image.shape == (horizon, 3, 96, 96)
    assert 0 <= image.min() <= image.max() <= 1
    assert sample["obs"]["agent_pos"].shape == (horizon, 2)
    assert sample["action"].shape == (horizon, 2)
    assert sample["action"].dtype == torch.float32
    assert torch.equal(sample["action"][0], sample["action"][1])
