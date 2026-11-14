import json
import threading
import time
import uuid

from real_robot.io_workers import AsyncFileRPC, AsyncJSONLines


def test_slow_image_serialization_never_blocks_actuator_queue_poll(tmp_path):
    entered,release=threading.Event(),threading.Event()
    def prepare(request):
        entered.set()
        assert release.wait(3)
        return request
    rpc=AsyncFileRPC(tmp_path,prepare)
    try:
        identifier=uuid.uuid4().hex
        (tmp_path/'model-request.json').write_text(json.dumps({'id':identifier,'command':'sample'}))
        assert entered.wait(2)
        began=time.perf_counter()
        assert rpc.pop() is None
        assert time.perf_counter()-began<.1
        release.set()
        item=rpc.pending.get(timeout=2)
        assert item['id']==identifier
        rpc.respond(item,ok=True)
        deadline=time.perf_counter()+2
        while not (tmp_path/(identifier+'.response.json')).exists() and time.perf_counter()<deadline:
            time.sleep(.002)
        assert json.loads((tmp_path/(identifier+'.response.json')).read_text())['ok']
        assert rpc.pop() is None
    finally:
        release.set()
        rpc.close()


def test_restart_does_not_replay_existing_rpc_request(tmp_path):
    old=uuid.uuid4().hex
    (tmp_path/'model-request.json').write_text(json.dumps({'id':old}))
    calls=[]
    rpc=AsyncFileRPC(tmp_path,lambda request:calls.append(request))
    try:
        time.sleep(.03)
        assert not calls
        new=uuid.uuid4().hex
        (tmp_path/'model-request.json').write_text(json.dumps({'id':new}))
        deadline=time.perf_counter()+2
        while not calls and time.perf_counter()<deadline:time.sleep(.002)
        assert [r['id'] for r in calls]==[new]
    finally:rpc.close()


def test_background_event_log_drains_every_row_in_order(tmp_path):
    logs=AsyncJSONLines(tmp_path)
    for index in range(300):
        logs.submit('sequence-actions',{'id':index,'target':[index]*7})
        logs.submit('manual-events',{'id':index})
    logs.close()
    assert logs.finalized.is_set() and logs.error is None
    assert logs.written==logs.submitted==600
    for name in ['sequence-actions','manual-events']:
        rows=[json.loads(s) for s in (tmp_path/(name+'.jsonl')).read_text().splitlines()]
        assert [r['id'] for r in rows]==list(range(300))
