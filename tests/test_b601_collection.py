"""Exercise the actual collection loop without constructing any hardware driver."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


spec = importlib.util.spec_from_file_location(
    "b601_collection", Path(__file__).resolve().parents[1] / "collection/b601/b601_pick_place.py")
collection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection)


def run_loop(monkeypatch, keys, next_output=None):
    state = np.array([-1., -80., -60., 45., -4., -17., -60.])
    names = (*collection.JOINTS, "gripper")
    directions = dict(zip(names, [-1., -1., 1., 1., 1., -1., -6.]))
    sent_inputs, returned_targets, datasets, saved = [], [], [], []
    clock = iter(range(1_000_000, 2_000_000, 1000))

    class Robot:
        config = SimpleNamespace(joint_directions=directions, joint_limits={"gripper": (-270., 0.)})

        def send_action(self, action):
            sent_inputs.append(action.copy())
            # A different limit deliberately makes returned targets differ from requests.
            target = {feature: action[feature] * directions[name]
                      for name, feature in zip(names, collection.FEATURE_NAMES)}
            target["gripper.pos"] = float(np.clip(target["gripper.pos"], state[-1]-7, state[-1]+7))
            returned_targets.append(target.copy())
            return target

    class Dataset:
        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)

    def create_dataset(root, image):
        dataset = Dataset()
        datasets.append(dataset)
        return dataset

    def save_episode(dataset, root, timings):
        saved.append((dataset, root, list(timings)))

    monkeypatch.setattr(collection, "read_state", lambda robot, tracking=None: state.copy())
    monkeypatch.setattr(collection, "read_camera", lambda camera: (np.zeros((8, 8, 3), np.uint8), next(clock)))
    monkeypatch.setattr(collection, "check_pose", lambda *args: np.zeros(3))
    monkeypatch.setattr(collection, "create_dataset", create_dataset)
    monkeypatch.setattr(collection, "save_episode", save_episode)
    monkeypatch.setattr(collection.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(collection.time, "monotonic", lambda: 0.)
    monkeypatch.setattr(collection.time, "sleep", lambda delay: None)
    key_stream = iter(keys)
    collection.control_loop(lambda: next(key_stream), lambda view: None,
                            Robot(), object(), SimpleNamespace(pose=lambda joints: np.eye(4)),
                            np.array([[-1., 1.]]*3), np.array([[-270., 270.]]*7),
                            Path("unused-episode-1"), next_output=next_output)
    return state, sent_inputs, returned_targets, datasets, saved


def test_recorder_stores_returned_targets_and_holds_them(monkeypatch):
    state, inputs, returned, datasets, saved = run_loop(monkeypatch, ["r", "p", "", "o", "\n", "q"])
    frames = datasets[0].frames
    assert len(frames) == 4 and len(saved) == 1
    np.testing.assert_array_equal(frames[0]["action"], state)
    assert frames[1]["action"][-1] == returned[0]["gripper.pos"] == -53
    np.testing.assert_array_equal(frames[2]["action"], frames[1]["action"])
    assert frames[3]["action"][-1] == returned[1]["gripper.pos"] == -67
    assert np.isclose(inputs[0]["gripper.pos"], -52 / -6)
    assert inputs[0]["shoulder_pan.pos"] == state[0] / -1
    timings = saved[0][2]
    assert [row[3] is not None for row in timings] == [False, True, False, True]
    for row in timings:
        assert row[1] < row[2]
        if row[3] is not None:
            assert row[2] < row[3]
    for frame in frames:
        np.testing.assert_array_equal(frame["observation.state"], state)


def test_next_recording_preserves_session_target(monkeypatch):
    state, _, returned, datasets, saved = run_loop(
        monkeypatch, ["r", "p", "", "\n", "r", "", "\n", "q"],
        next_output=lambda: Path("unused-episode-2"))
    assert len(datasets) == len(saved) == 2
    first_next = datasets[1].frames[0]
    np.testing.assert_array_equal(first_next["observation.state"], state)
    np.testing.assert_array_equal(first_next["action"], datasets[0].frames[-1]["action"])
    assert first_next["action"][-1] == returned[0]["gripper.pos"] == -53
    assert saved[1][2][0][0] == 0
    assert saved[1][2][0][3] is None
