#!/usr/bin/env python3
"""Execute Lecture 13's actual task methods with CPU torch and explicit fake physics.

No Isaac constructor, data-view, collision, or forward-kinematics behavior is tested.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace as NS


def experiment():
    import torch
    lesson = Path(__file__).resolve().parents[1]/"Lecture-13.md"
    blocks = re.findall(r"```python\n(.*?)\n```", lesson.read_text(), re.S)
    contract_code = next(code for code in blocks if "def integrate_target(" in code)
    contract = {}
    exec(compile(contract_code, str(lesson), "exec"), contract)
    block = next(code for code in blocks if "class PushEnv(" in code)
    definition = next(node for node in ast.parse(block).body if isinstance(node, ast.ClassDef))

    class FakeBase:
        def _reset_idx(self, ids): self.episode_length_buf[ids] = 0

    namespace = {"DirectRLEnv": FakeBase, "torch": torch, "math": math, "C": NS(**contract),
                 # Deliberately deterministic fixture sampler: always choose the upper endpoint.
                 "sample_uniform": lambda lo,hi,shape,dev: torch.full(shape,hi,device=dev)}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(lesson),"exec"),namespace)
    env = object.__new__(namespace["PushEnv"])
    env.device,env.num_envs,env.step_dt,env.max_episode_length = "cpu",3,1/60,360
    env.q0 = torch.zeros(3,9);env.q0[:,:7]=.99
    env.q_lo,env.q_hi = -torch.ones(3,7),torch.ones(3,7)
    env.target,env.last_a,env.prev_applied = torch.zeros(3,7),torch.zeros(3,7),torch.zeros(3,7)
    env.hist,env.delay = torch.zeros(3,4,7),torch.tensor([2,0,1])
    env.goal = torch.tensor([[.5,0,.025]]*3)
    env.last_episode_success = torch.zeros(3,dtype=torch.bool)
    env.episode_length_buf = torch.tensor([359,359,359])
    env.cfg = NS(w_reach=1.,w_push=4.,w_success=10.,w_rate=-.01,w_qd=-1e-4,
                 success_tol=.03,cube_xy_range=((.4,.6),(-.15,.15)),goal_offset_range=(.1,.2),
                 min_action_delay=0,max_action_delay=2)
    env.scene = NS(env_origins=torch.tensor([[0.,0,0],[2.,0,0],[4.,0,0]]))
    data = NS(joint_pos=NS(torch=env.q0.clone()),joint_vel=NS(torch=torch.zeros(3,9)))
    env.robot = NS(data=data,_ALL_INDICES=torch.arange(3))
    def position(*,position,env_ids): data.joint_pos.torch[env_ids]=position
    def velocity(*,velocity,env_ids): data.joint_vel.torch[env_ids]=velocity
    env.robot.write_joint_position_to_sim_index=position
    env.robot.write_joint_velocity_to_sim_index=velocity
    cube_position = env.goal.clone()
    def root_pose(*,root_pose,env_ids):
        cube_position[env_ids]=root_pose[:,:3]-env.scene.env_origins[env_ids]
    env.cube=NS(write_root_pose_to_sim_index=root_pose,
                write_root_velocity_to_sim_index=lambda **kwargs:None)
    # Poses are explicit inputs. This replaces only the simulator-dependent frame reader.
    env._frames=lambda:(cube_position.clone(),cube_position.clone())
    refresh=[];env.sim=NS(forward=lambda:refresh.append("kinematics requested"))

    env._pre_physics_step(torch.ones(3,7))
    assert env.last_a[:,0].tolist()==[0.,1.,0.]
    assert torch.allclose(env.target[:,0],torch.tensor([0.,.05,0.]))
    assert env._get_observations()["policy"].shape==(3,31)
    assert torch.equal(env._get_observations()["policy"][:,-7:],env.last_a)
    assert torch.allclose(env._get_rewards(),torch.tensor([15.,14.93,15.])/60)
    env._pre_physics_step(-torch.ones(3,7))
    assert env.last_a[:,0].tolist()==[0.,-1.,1.]
    env._pre_physics_step(torch.zeros(3,7))
    assert env.last_a[:,0].tolist()==[1.,0.,-1.]
    env.delay[:]=0;env.target[:]=.99
    env._pre_physics_step(torch.ones(3,7))
    assert torch.equal(env.target,torch.ones(3,7))
    lost,timeout=env._get_dones();assert not lost.any() and not timeout.any()
    env.episode_length_buf+=1
    _,timeout=env._get_dones();assert timeout.all()
    cube_position[0,2]=-.06
    lost,_=env._get_dones();assert lost.tolist()==[True,False,False]
    before=[value.clone() for value in (env.target,env.last_a,env.prev_applied,env.hist)]
    env._reset_idx(torch.tensor([1]))
    for value,old in zip((env.target,env.last_a,env.prev_applied,env.hist),before):
        assert torch.equal(value[[0,2]],old[[0,2]])
    assert not env.last_a[1].any() and not env.prev_applied[1].any() and not env.hist[1].any()
    assert torch.equal(env.target[1],torch.ones(7)), "reset jitter must respect soft limits"
    assert env.last_episode_success.tolist()==[False,True,False]
    assert env.episode_length_buf.tolist()==[360,0,360] and len(refresh)==1
    return {"scope":"actual CPU PyTorch execution of lecture class methods; fake scene/poses/kinematics",
            "torch":torch.__version__,"checks_passed":8,
            "checks":["per-environment delay buffer", "31-value observation uses applied action",
                      "known action-rate reward", "soft target limits", "timeout at 360 transitions",
                      "lost-cube termination", "selected reset/history/success isolation",
                      "reset clipping and kinematics request"],
            "task_code_sha256":hashlib.sha256(block.encode()).hexdigest(),
            "contract_code_sha256":hashlib.sha256(contract_code.encode()).hexdigest(),
            "limitations":["constructor and Isaac asset/data APIs not executed", "poses and kinematics are fixtures",
                           "no training, physics, renderer, or learned task success"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    encoded=json.dumps(experiment(),indent=2,allow_nan=False)+"\n"
    if args.output:args.output.write_text(encoded)
    print(encoded,end="")
