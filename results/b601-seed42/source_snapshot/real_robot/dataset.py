"""Seven-joint B601 images and motor-degree targets, sampled within episodes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(selected, n_validation=4, seed=42):
    selected = [int(index) for index in selected]
    if len(set(selected)) != len(selected) or any(index < 0 for index in selected):
        raise ValueError("Episode indices must be unique and nonnegative")
    if not 0 < n_validation < len(selected):
        raise ValueError("Both training and validation need at least one episode")
    validation = sorted(np.random.default_rng(seed).choice(selected, size=n_validation, replace=False).tolist())
    return {
        "seed": seed, "selected_episode_indices": sorted(selected),
        "train_episode_indices": sorted(set(selected)-set(validation)),
        "validation_episode_indices": validation,
    }


def write_split(selection_path, episodes_path, output_path, seed=42, n_validation=4):
    selection = json.loads(Path(selection_path).read_text())
    split = make_split(selection["candidate_episode_indices"], n_validation, seed)
    split.update(selection_sha256=sha256(selection_path), episodes_sha256=sha256(episodes_path),
                 selection_path=str(selection_path), episodes_path=str(episodes_path))
    output_path = Path(output_path)
    if output_path.exists():
        if json.loads(output_path.read_text()) != split:
            raise ValueError(f"Existing split differs: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x") as handle:
            handle.write(json.dumps(split, indent=2)+"\n")
    return split


class B601ImageDataset(BaseImageDataset):
    def __init__(self, zarr_path, selection_path, split_path, horizon=16,
                 n_obs_steps=2, pad_before=1, pad_after=7):
        super().__init__()
        if not 1 <= n_obs_steps <= horizon:
            raise ValueError("Observation history must fit within the sequence")
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        self.selection_path, self.split_path = str(selection_path), str(split_path)
        self.split = json.loads(Path(split_path).read_text())
        selection = json.loads(Path(selection_path).read_text())
        if self.split["selection_sha256"] != sha256(selection_path):
            raise ValueError("Dataset selection changed after the split was frozen")
        episodes_path = Path(zarr_path).parent / "episodes.json"
        if self.split["episodes_sha256"] != sha256(episodes_path):
            raise ValueError("Episode provenance changed after the split was frozen")
        selected = set(selection["candidate_episode_indices"])
        train = set(self.split["train_episode_indices"])
        validation = set(self.split["validation_episode_indices"])
        if train & validation or train | validation != selected or not train or not validation:
            raise ValueError("Training and validation must partition selected episodes")
        self.episode_ends = self.replay_buffer.episode_ends[:]
        self.episode_starts = np.r_[0, self.episode_ends[:-1]]
        if not selected or min(selected) < 0 or max(selected) >= len(self.episode_ends):
            raise ValueError("Selected episode is outside the replay")
        if self.replay_buffer["state"].shape[1:] != (7,) or self.replay_buffer["action"].shape[1:] != (7,):
            raise ValueError("B601 requires all seven state and action features")
        self.train_mask = np.isin(np.arange(len(self.episode_ends)), sorted(train))
        self.validation_mask = np.isin(np.arange(len(self.episode_ends)), sorted(validation))
        self.episode_mask = self.train_mask
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.pad_before, self.pad_after = pad_before, pad_after
        self.sampler = self._sampler(self.episode_mask)
        self._normalizer = None

    def _sampler(self, mask):
        # Only sample low-dimensional arrays here. Images are read on demand for
        # the first observation steps, without copying the whole replay to RAM.
        return SequenceSampler(self.replay_buffer, sequence_length=self.horizon,
                               pad_before=self.pad_before, pad_after=self.pad_after,
                               keys=["state", "action"], episode_mask=mask)

    def get_validation_dataset(self):
        validation = copy.copy(self)
        validation.episode_mask = self.validation_mask
        validation.sampler = validation._sampler(validation.episode_mask)
        return validation

    def training_values(self, key):
        return np.concatenate([self.replay_buffer[key][start:end]
                               for start, end, keep in zip(self.episode_starts, self.episode_ends, self.train_mask)
                               if keep], axis=0)

    def get_normalizer(self, **kwargs):
        if self._normalizer is None:
            normalizer = LinearNormalizer()
            normalizer.fit({"action": self.training_values("action"),
                            "joint_pos": self.training_values("state")}, last_n_dims=1, **kwargs)
            normalizer["image"] = get_image_range_normalizer()
            self._normalizer = normalizer
        return self._normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.training_values("action"))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        start, end, sample_start, _ = self.sampler.indices[index]
        frame_indices = np.clip(start+np.arange(self.n_obs_steps)-sample_start, start, end-1)
        images = self.replay_buffer["img"].oindex[frame_indices]
        images = np.moveaxis(images, -1, 1).astype(np.float32) / 255.
        return {"obs": {"image": torch.from_numpy(np.ascontiguousarray(images)),
                        "joint_pos": torch.from_numpy(sample["state"][:self.n_obs_steps].astype(np.float32))},
                "action": torch.from_numpy(sample["action"].astype(np.float32))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze the B601 episode split")
    parser.add_argument("--selection", default="results/b601-data-check/review/training_candidates.json")
    parser.add_argument("--episodes", default="data/b601/merged/episodes.json")
    parser.add_argument("--output", default="data/b601/merged/split-seed42.json")
    args = parser.parse_args()
    print(json.dumps(write_split(args.selection, args.episodes, args.output), indent=2))
