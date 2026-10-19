#!/usr/bin/env python3
"""Execute Lecture 10's actual task methods with CPU torch and an explicit fake scene.

This checks tensor shapes/reward/reset logic, not Isaac APIs or FK correctness.
"""
import argparse
import ast
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS


def experiment():
    import torch
    lesson = Path(__file__).resolve().parents[1]/"Lecture-10.md"
    block = next(code for code in re.findall(r"```python\n(.*?)\n```", lesson.read_text(), re.S)
                 if "class FrankaReachEnv(" in code)
    parsed = ast.parse(block)
    definition = next(node for node in parsed.body if isinstance(node, ast.ClassDef))

    class FakeBase:
        def _reset_idx(self, ids): self.episode_length_buf[ids] = 0

    namespace = {"DirectRLEnv": FakeBase, "FrankaReachEnvCfg": object,
                 "torch": torch, "Sequence": Sequence}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(lesson), "exec"), namespace)
    env = object.__new__(namespace["FrankaReachEnv"])
    env.device, env.num_envs = "cpu", 3
    env.cfg = NS(action_scale=.5, rew_distance=-1., rew_near=1., rew_action_rate=-.01, success_radius=.05)
    env.step_dt, env.max_episode_length = 1/60, 240
    env.arm_ids, env.ee_idx = torch.arange(7), 0
    env.q0 = torch.zeros(3,7); env.q0[:,3] = -.9
    env.q_lo, env.q_hi = torch.full((3,7),-1.), torch.ones(3,7)
    env.q_targets = env.q0.clone()
    env.actions, env.prev_actions = torch.zeros(3,7), torch.zeros(3,7)
    env.targets, env.ee_pos = torch.zeros(3,3), torch.zeros(3,3)
    env.lo, env.hi = torch.tensor([.2,-.2,.1]), torch.tensor([.5,.2,.4])
    env.dist, env.success = torch.zeros(3), torch.zeros(3,dtype=torch.bool)
    env.episode_length_buf = torch.tensor([239,239,239])
    env.extras = {}
    origin = torch.tensor([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]])
    env.scene = NS(env_origins=origin)
    default_q = torch.cat((env.q0,torch.zeros(3,2)),dim=1)
    data = NS(default_joint_pos=NS(torch=default_q), joint_pos=NS(torch=default_q.clone()),
              joint_vel=NS(torch=torch.zeros(3,9)), body_pos_w=NS(torch=origin[:,None,:].clone()))
    env.robot = NS(data=data)
    def write_position(*,position,env_ids): data.joint_pos.torch[env_ids]=position
    def write_velocity(*,velocity,env_ids): data.joint_vel.torch[env_ids]=velocity
    env.robot.write_joint_position_to_sim_index=write_position
    env.robot.write_joint_velocity_to_sim_index=write_velocity
    refresh=[]
    env.sim=NS(forward=lambda:refresh.append("kinematics requested"))

    action=torch.zeros(3,7);action[:,3]=-1
    env._pre_physics_step(action)
    assert torch.all(env.q_targets[:,3] == -1), "soft limits must clamp q0+offset"
    _,timeout=env._get_dones()
    assert not timeout.any()
    env.episode_length_buf+=1
    _,timeout=env._get_dones()
    assert timeout.all(), "timeout at exactly 240 transitions, not 239 or 241"
    obs=env._get_observations()["policy"]
    assert obs.shape == (3,27) and torch.equal(obs[:,-7:],action)
    # At the goal, distance=0 and near reward=1; changing one action by -1 costs .01.
    assert torch.allclose(env._get_rewards(),torch.full((3,),.99/60))
    before=env.actions.clone();target_before=env.q_targets.clone()
    env._reset_idx(torch.tensor([1]))
    assert torch.equal(env.actions[[0,2]],before[[0,2]])
    assert torch.equal(env.q_targets[[0,2]],target_before[[0,2]])
    assert not env.actions[1].any() and not env.prev_actions[1].any()
    assert torch.equal(env.q_targets[1],env.q0[1]) and len(refresh)==1
    assert env.episode_length_buf.tolist()==[240,0,240]
    return {"scope":"actual CPU PyTorch execution of lecture class methods; fake scene and kinematics callback",
            "torch":torch.__version__,"observation_width":obs.shape[1],"checks_passed":6,
            "checks":["soft joint-limit clipping","timeout at 240 transitions","previous action in observation",
                      "known reward .99/60","selected reset only","kinematics refresh requested"],
            "task_code_sha256":hashlib.sha256(block.encode()).hexdigest(),
            "limitations":["constructor/asset loading not executed","no Isaac data views or forward kinematics",
                            "real reset sensor freshness requires simulator execution"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    encoded=json.dumps(experiment(),indent=2,allow_nan=False)+"\n"
    if args.output: args.output.write_text(encoded)
    print(encoded,end="")
