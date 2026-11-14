"""Explicitly enable the B601 at its measured pose; maintain only that pose."""
import argparse,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from motorbridge import Controller,Mode

parser=argparse.ArgumentParser()
parser.add_argument('--setup',type=Path,required=True)
args=parser.parse_args(); setup=json.loads(args.setup.read_text(encoding='utf-8'))
out=args.setup.parent; status_file=out/'hold-status.json'; control=out/'hold-command.json'
controller=None; motors=[]; enabled=False; targets=None

def save(value):
    value.update(pid=os.getpid(),updated_ns=time.perf_counter_ns(),model_trajectory_commands=0)
    try:
        status_file.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    except OSError as error:
        print('Status write failed:',repr(error),file=sys.stderr,flush=True)

def read():
    for motor in motors:motor.request_feedback()
    for _ in range(3):time.sleep(.01);controller.poll_feedback_once()
    states=[m.get_state() for m in motors]
    if any(s is None for s in states):raise RuntimeError('Missing motor feedback')
    q=np.array([math.degrees(s.pos) for s in states])
    if not np.isfinite(q).all():raise RuntimeError('Nonfinite motor feedback')
    return states,q

def xyz(q):
    T=np.eye(4)
    for degree,origin,axis in zip(q[:6],setup['origins'],setup['axes']):
        T=T@np.array(origin);a=np.array(axis);a=a/np.linalg.norm(a);x,y,z=a
        K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);angle=math.radians(degree)
        R=np.eye(3)+math.sin(angle)*K+(1-math.cos(angle))*(K@K)
        T[:3,:3]=T[:3,:3]@R
    return (T@np.array(setup['tool']))[:3,3]

def hold():
    for i,(motor,target) in enumerate(zip(motors,targets)):
        if i<6:motor.send_pos_vel(float(target),math.radians(5))
        else:motor.send_force_pos(float(target),math.radians(3),.1)

try:
    controller=Controller.from_dm_serial(setup['port'],921600)
    motors=[controller.add_damiao_motor(*ids,'4340P' if i<3 else '4310') for i,ids in enumerate(setup['motor_ids'])]
    states,q=read();position=xyz(q);bounds=np.array(setup['bounds']);limits=np.array(setup['limits'])
    if np.any(q<limits[:,0]-.1) or np.any(q>limits[:,1]+.1):raise RuntimeError('Current pose outside joint limits')
    if np.any(position<bounds[:,0]) or np.any(position>bounds[:,1]):raise RuntimeError('Current tool position outside workspace; motors not enabled')
    targets=np.array([s.pos for s in states])
    save({'state':'preparing_current_pose_hold','current_motor_degrees':q.tolist(),'xyz_m':position.tolist()})
    already_enabled=all(s.status_code==1 for s in states)
    if not already_enabled:
        if any(s.status_code!=0 for s in states):raise RuntimeError('Unexpected motor states; not changing modes')
        for i,motor in enumerate(motors):motor.ensure_mode(Mode.POS_VEL if i<6 else Mode.FORCE_POS,timeout_ms=1000)
    # Preload measured positions and conservative velocity limits before enabling.
    hold()
    if not already_enabled:controller.enable_all()
    enabled=True;hold()
    while True:
        started=time.perf_counter();hold();states,q=read()
        command=None
        if control.exists():
            command=json.loads(control.read_text(encoding='utf-8')).get('command');control.unlink()
        if command=='release':
            controller.disable_all();enabled=False
            save({'state':'released','motor_status_codes':[s.status_code for s in states]});os._exit(0)
        if command=='hold_current':targets=np.array([s.pos for s in states]);hold()
        error=float(np.max(np.abs(q-np.rad2deg(targets))))
        if error>2.0:
            targets=np.array([s.pos for s in states]);hold()
            save({'state':'holding_measured_pose_after_tracking_error','tracking_error_before_hold_deg':error,'current_motor_degrees':q.tolist(),'motor_status_codes':[s.status_code for s in states]})
        else:
            save({'state':'holding','current_motor_degrees':q.tolist(),'hold_motor_degrees':np.rad2deg(targets).tolist(),'max_tracking_error_deg':error,'motor_status_codes':[s.status_code for s in states],'xyz_m':xyz(q).tolist()})
        time.sleep(max(0,.1-(time.perf_counter()-started)))
except BaseException as error:
    if enabled and targets is not None:
        try:hold()
        except BaseException:pass
    save({'state':'error','error':repr(error),'motors_were_enabled':enabled})
    os._exit(1)
