import numpy as np
import pytest

from real_robot.receding_horizon import DT_NS, TimedActionQueue


def test_sequence_executes_at_ten_hz_without_waiting_for_arrival():
    queue = TimedActionQueue()
    actions = np.arange(15)[:,None]*np.ones((1,7))
    times = 1_000_000_000+np.arange(15)*DT_NS
    queue.replace(actions, times, 950_000_000, "first")
    assert queue.due(999_999_999) is None
    for index, timestamp in enumerate(times):
        point = queue.due(int(timestamp)+5_000_000)
        assert point["action_index"] == index
        np.testing.assert_array_equal(point["target"], actions[index])
        assert queue.due(int(timestamp)+10_000_000) is None


def test_replanning_keeps_old_prefix_and_replaces_future_without_replaying_stale_actions():
    queue = TimedActionQueue()
    actions = np.zeros((15,7));times = 1_000_000_000+np.arange(15)*DT_NS
    queue.replace(actions, times, 950_000_000, "old")
    for timestamp in times[:8]:
        queue.due(int(timestamp))
    new_times = times+660_000_000
    accepted = queue.replace(actions+1, new_times, 1_750_000_000, "new")
    assert accepted["discarded_expired_actions"] == 2
    assert queue.due(1_800_000_000)["sequence_id"] == "old"
    assert queue.due(1_860_000_000)["sequence_id"] == "new"
    assert all(p["sequence_id"] == "new" for p in queue.points)
    queue.clear()
    assert queue.due(9_000_000_000) is None


def test_submillisecond_timestamp_jitter_does_not_queue_two_targets_in_one_slot():
    queue=TimedActionQueue();actions=np.zeros((15,7));times=1_000_000_000+np.arange(15)*DT_NS
    queue.replace(actions,times,950_000_000,"old")
    for timestamp in times[:7]:queue.due(int(timestamp))
    queue.replace(actions+1,times+600_236_000,1_650_000_000,"new")
    assert queue.due(1_700_000_000) is None
    point=queue.due(1_720_000_000)
    assert point["sequence_id"]=='new'
    assert queue.due(1_750_000_000) is None


def test_stale_plan_bad_timing_and_dispatch_overrun_are_rejected():
    queue = TimedActionQueue();actions=np.zeros((15,7));times=1_000_000_000+np.arange(15)*DT_NS
    with pytest.raises(ValueError, match="过期"):
        queue.replace(actions,times,3_000_000_000,"stale")
    with pytest.raises(ValueError, match="10 Hz"):
        queue.replace(actions,times+np.arange(15),0,"wrong")
    queue.replace(actions,times,950_000_000,"valid")
    with pytest.raises(ValueError, match="迟到"):
        queue.due(1_076_000_000)


def test_per_run_action_count_excludes_previous_runs(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.b601_rhc_loop import run_rhc
    (tmp_path/'teleop-status.json').write_text(json.dumps({'sequence_actions_sent':42,'last_fault':None}))
    trial=SimpleNamespace(request=lambda *args,**kwargs:('stop',{'sequence_actions_sent':50}))
    args=SimpleNamespace(session=tmp_path,duration=0,recording_id='test',hold_validation=False)
    result={'run_id':'new'}
    run_rhc(args,trial,None,None,result,None,None,lambda:None)
    assert result['sequence_action_counter_start']==42
    assert result['completed_steps']==8
