import importlib.util
from pathlib import Path

import numpy as np
import pytest

from real_robot.manual_control import (GRIPPER_KEYS, MOVES, GripperProgress,
                                      JogModel, KeyLatch, check_arm_tracking)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manual_reference", ROOT/"collection/b601/b601_pick_place.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture
def model():
    original = reference.Kinematics(reference.URDF)
    return JogModel({"origins": [value.tolist() for value in original.origins],
                     "axes": [value.tolist() for value in original.axes], "tool": original.tool.tolist(),
                     "limits": original.limits_deg.tolist()+[[-270., 0.]],
                     "bounds": [[.19, .38], [-.09, .113], [.004, .152]]})


def test_portable_model_matches_original_for_nonzero_joint_rotations(model):
    original = reference.Kinematics(reference.URDF)
    for q in ([0.]*6, [3., -98., -59., 52., 3., -4.], [-15., -110., -70., 48., -4., -17.]):
        np.testing.assert_allclose(model.pose(q), original.pose(q), rtol=0, atol=1e-12)


@pytest.mark.parametrize("key", list(MOVES))
def test_each_translation_key_requests_six_mm_and_preserves_orientation(model, key):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    start = model.pose(current)
    target, destination = model.goal(current, key)
    np.testing.assert_allclose(destination-start[:3, 3], .006*np.array(MOVES[key]), atol=1e-12)
    np.testing.assert_allclose(model.pose(target)[:3, 3], destination, atol=.0001)
    np.testing.assert_allclose(model.pose(target)[:3, :3], start[:3, :3], atol=.001)
    assert target[6] == current[6]


def test_workspace_violation_rejects_entire_step_without_shortening(model):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    x = model.pose(current)[0, 3]
    model.bounds[0, 1] = x+.003
    with pytest.raises(ValueError, match="6 mm 目标越界"):
        model.goal(current, "w")


def test_left_opens_and_right_closes_in_motor_units(model):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    for key, sign in GRIPPER_KEYS.items():
        target, _ = model.goal(current, key)
        np.testing.assert_array_equal(target[:6], current[:6])
        assert target[6] == current[6]+sign*8
    current[6] = 0.
    with pytest.raises(ValueError, match="夹爪已到"):
        model.goal(current, "Right")


def test_repeated_keydown_requires_release_before_next_step():
    latch = KeyLatch()
    assert latch.press("w")
    assert not any(latch.press("w") for _ in range(20))
    latch.release("w")
    assert latch.press("w")


def test_gripper_normal_travel_does_not_trigger_arm_tracking_guard():
    current = np.array([3., -98., -59., 52., 3., -4., -1.8])
    last_sent = current.copy()
    last_sent[6] = -8.
    check_arm_tracking(current, last_sent)
    progress = GripperProgress(0., -8., 0.)
    for position, now in [(-1.8, .5), (-4., 1.), (-6., 1.5)]:
        assert not progress.update(position, now)
    assert progress.update(-7.9, 2.)


def test_arm_tracking_protection_is_preserved():
    current, sent = np.zeros(7), np.zeros(7)
    sent[2] = 2.1
    with pytest.raises(ValueError, match="机械臂关节 3"):
        check_arm_tracking(current, sent)


def test_gripper_stall_and_motion_outside_requested_interval_stop():
    with pytest.raises(ValueError, match="无开合进展"):
        GripperProgress(0., -8., 0.).update(0., 1.6)
    with pytest.raises(ValueError, match="超出本次开合行程"):
        GripperProgress(0., -8., 0.).update(-9., .5)
    with pytest.raises(ValueError, match="单次开合目标无效"):
        GripperProgress(0., -20., 0.)


def test_bounded_model_step_preserves_direction_and_checks_entire_path(model):
    from real_robot.model_control import bounded_target, validate_joint_path
    current = np.array([3., -98., -59., 52., 3., -4., -80.])
    delta = np.array([10., -2., 3., 4., -1., 2., 30.])
    target = bounded_target(model, current, current+delta)
    assert np.max(np.abs(target[:6]-current[:6])) <= .500001
    assert abs(target[6]-current[6]) <= 2.000001
    np.testing.assert_allclose((target-current)/delta, np.full(7, (target[0]-current[0])/delta[0]))
    assert np.linalg.norm(model.pose(target)[:3, 3]-model.pose(current)[:3, 3]) <= .006
    with pytest.raises(ValueError, match="变化超限"):
        validate_joint_path(model, current, current+delta, max_delta=[.5]*6+[2.])
    model.bounds[0, 1] = model.pose(current)[0, 3]-.001
    with pytest.raises(ValueError, match="工作区"):
        bounded_target(model, current, current+delta)


def test_model_observation_rejects_stale_moved_and_nonfinite_feedback():
    from real_robot.model_control import validate_observation
    q = np.zeros(7)
    validate_observation(q, q, 100, 200)
    with pytest.raises(ValueError, match="过期"):
        validate_observation(q, q, 100, 1_000_000_101)
    with pytest.raises(ValueError, match="移动"):
        validate_observation(q+.31, q, 100, 200)
    with pytest.raises(ValueError):
        validate_observation(q+np.nan, q, 100, 200)


def test_native_model_request_requires_matching_controller_and_idle_state(model, tmp_path, request):
    import json
    import os
    import time
    import uuid
    import threading
    from types import SimpleNamespace
    from scripts.b601_model_camera import ExternalControl
    from real_robot.io_workers import AsyncFileRPC

    current = np.array([3., -98., -59., 52., 3., -4., -80.])
    control = ExternalControl.__new__(ExternalControl)
    control.directory, control.seen = tmp_path, set()
    control.active, control.executed, control.policy_executed = None, 0, 0
    control.recorder = None
    from real_robot.receding_horizon import TimedActionQueue
    control.action_queue, control.rhc_running = TimedActionQueue(), False
    control.hardware = SimpleNamespace(model=model, busy=False, paused=True, current=current, target=current,
                                       control_generation=0, setup={}, lock=threading.Lock(),feedback_ns=time.perf_counter_ns())
    control.rpc=AsyncFileRPC(tmp_path,control.prepare_request)
    request.addfinalizer(control.rpc.close)

    def submit(**overrides):
        request = {"id": uuid.uuid4().hex, "command": "execute", "controller_pid": os.getpid(),
                   "control_generation": 0,
                   "issued_unix": time.time(), "controller_lease_ns": time.perf_counter_ns(),
                   "observed_ns": time.perf_counter_ns(),
                   "observed_q": current.tolist(), "target": (current+.1).tolist(), **overrides}
        (tmp_path/"model-request.json").write_text(json.dumps(request))
        return request

    def drive(request):
        deadline=time.perf_counter()+2
        while time.perf_counter()<deadline:
            goal=control.poll(current)
            if goal is not None:
                return goal
            if (tmp_path/(request['id']+'.response.json')).exists():
                return None
            time.sleep(.002)
        raise AssertionError('No response')

    def response(request):
        path=tmp_path/(request['id']+'.response.json')
        deadline=time.perf_counter()+2
        while time.perf_counter()<deadline:
            try:return json.loads(path.read_text())
            except (OSError,ValueError):time.sleep(.002)
        raise AssertionError('No response')

    rejected = submit(controller_pid=-1)
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    control.hardware.busy = True
    rejected = submit()
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    control.hardware.busy = False
    rejected = submit(control_generation=-1)
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    rejected = submit(controller_lease_ns=time.perf_counter_ns()-3_100_000_000)
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    control.hardware.setup["full_recording"] = True
    rejected = submit()
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    control.hardware.setup["full_recording"] = False
    control.hardware.setup["align_reference_on_start"]=False
    rejected=submit(purpose="reference_alignment")
    assert drive(rejected) is None
    assert not response(rejected)['ok']
    queued=submit()
    deadline=time.perf_counter()+2
    while control.rpc.pending.empty() and time.perf_counter()<deadline:
        time.sleep(.002)
    assert not control.rpc.pending.empty()
    control.hardware.control_generation=1
    assert drive(queued) is None
    assert not response(queued)['ok']
    control.hardware.control_generation=0
    accepted = submit()
    np.testing.assert_allclose(drive(accepted), current+.1)
    assert control.poll(current) is None  # the same request never executes twice
    control.finish("操作员暂停")
    assert not response(accepted)['ok']
    assert control.executed == 0


def test_dispatch_feedback_jitter_is_bounded_again_without_relaxing_limits(model):
    from real_robot.model_control import prepare_execution
    observed = np.array([3., -98., -59., 52., 3., -4., -80.])
    current, target = observed.copy(), observed.copy()
    current[0] -= .03
    target[0] += .5
    accepted = prepare_execution(model, current, observed, target, 100, 200)
    assert accepted[0]-current[0] == pytest.approx(.5)
    assert accepted[0] < target[0]
    with pytest.raises(ValueError, match="变化超限"):
        prepare_execution(model, current, observed, target+.6, 100, 200)


def test_operator_pause_invalidates_the_continuous_run_even_between_actions():
    import threading
    from scripts.b601_manual_control import Hardware
    hardware = Hardware.__new__(Hardware)
    hardware.lock = threading.Lock()
    hardware.pause_event = threading.Event()
    hardware.control_generation = 7
    hardware.pending = "w"
    hardware.paused = False
    hardware.pause()
    assert hardware.control_generation == 8
    assert hardware.paused and hardware.pause_event.is_set() and hardware.pending is None


def test_runner_writes_no_further_request_after_operator_pause(tmp_path):
    import json
    from scripts.run_b601_model_trial import Trial
    status = {"model_control_enabled": True, "connected": True, "pid": 123,
              "control_generation": 4, "updated_ns": 100}
    path = tmp_path/"teleop-status.json"
    path.write_text(json.dumps(status))
    runner = Trial(tmp_path, control_generation=4)
    status["control_generation"] = 5
    path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="操作员已暂停"):
        runner.request("sample")
    assert not (tmp_path/"model-request.json").exists()
    with pytest.raises(RuntimeError, match="操作员已暂停"):
        Trial(tmp_path, control_generation=4)


def test_rhc_validates_full_future_path_without_scaling_predictions(model):
    from real_robot.receding_horizon import validate_sequence
    q = np.array([3., -98., -59., 52., 3., -4., -80.])
    actions = np.tile(q,(15,1));actions[:,0] += np.arange(15)*.05
    np.testing.assert_array_equal(validate_sequence(model,q,actions),actions)
    bad=actions.copy();bad[7,2] += 3.
    with pytest.raises(ValueError, match="单步范围"):
        validate_sequence(model,q,bad)
    model.bounds[0,1]=model.pose(q)[0,3]-.001
    with pytest.raises(ValueError, match="工作区"):
        validate_sequence(model,q,actions)


def test_rhc_first_gripper_goal_uses_collector_clipping_instead_of_rejecting_absolute_goal(model):
    from real_robot.receding_horizon import validate_sequence,project_dispatch_target
    q = np.array([3., -98., -59., 52., 3., -4., -104.5])
    actions = np.tile(q,(15,1));actions[:,6] += 8.62
    np.testing.assert_array_equal(validate_sequence(model,q,actions),actions)
    target = project_dispatch_target(model,q,actions[0])
    assert target[6] == pytest.approx(q[6]+8.)
    np.testing.assert_array_equal(target[:6],actions[0,:6])
