"""Step 5: ACT end-effector-pose policy executed through IK on the Piper.

Loads a trained ACT policy (EE-pose action head), turns each predicted
[ee_pos(3), rot6d(6), gripper(1)] into a 6-joint reference via Piper IK, and
passes the gripper through. This is the glue that lets a pose-space VLA/ACT act
on the Piper, the paper's "policy acts through IK" interface.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch

from .piper_fk import PiperFK, rot6d_to_mat
from .piper_ik import PiperIK


def _load_act(ckpt: str):
    err = None
    for mod in ("lerobot.policies.act.modeling_act",
                "lerobot.common.policies.act.modeling_act"):
        try:
            ACTPolicy = importlib.import_module(mod).ACTPolicy
            return ACTPolicy.from_pretrained(ckpt)
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"could not load ACTPolicy: {err}")


class ACTtoJoint:
    """Run ACT and convert its EE-pose output to Piper joint references via IK.

    Uses LeRobot 0.7's processor pipeline (input normalization + output
    denormalization) exactly as ``lerobot_eval`` does -- without it the policy
    sees/returns raw values and predicts garbage.
    """

    def __init__(self, ckpt: str, device: str = "cuda", model_path: str | None = None) -> None:
        from lerobot.policies import make_pre_post_processors

        self.policy = _load_act(ckpt).to(device).eval()
        self.device = device
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.fk = PiperFK(model_path) if model_path else PiperFK()
        self.ik = PiperIK(self.fk)

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def predict_pose(self, obs: dict):
        """ACT forward with normalization: returns the 10-D EE-pose action."""
        from lerobot.processor import policy_output_to_transition, transition_to_policy_action

        batched = {}
        for k, v in obs.items():
            batched[k] = v.unsqueeze(0) if torch.is_tensor(v) else v
        task = obs.get("task", "")
        batched["task"] = task if isinstance(task, (list, tuple)) else [task]

        proc = self.preprocessor(batched)
        out = self.policy.select_action(proc)
        action = transition_to_policy_action(policy_output_to_transition(out))
        action = self.postprocessor(action)
        return np.asarray(action.squeeze(0).float().cpu().numpy())

    def act(self, obs: dict, q_current: np.ndarray):
        """obs -> (q6 joint reference, gripper, goal_pose(pos,R), ik_residual(pos,rot))."""
        a = self.predict_pose(obs)
        pos, r6, grip = a[:3], a[3:9], float(a[9])
        R = rot6d_to_mat(r6)
        q, e_p, e_r = self.ik.solve(pos, R, q_current)
        return q, grip, (pos, R), (e_p, e_r)
