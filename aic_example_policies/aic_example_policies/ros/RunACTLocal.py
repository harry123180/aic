#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import time
import json
import torch
import numpy as np
import cv2
import draccus
from pathlib import Path
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from safetensors.torch import load_file

CHECKPOINT_PATH = Path(
    "/home/harry/training/act_cheatcode/checkpoints/200000/pretrained_model"
)

# Limit maximum position delta per step to avoid sudden large movements
MAX_POSITION_DELTA = 0.02  # metres


class RunACTLocal(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------ #
        # 1. Load config and weights from local checkpoint
        # ------------------------------------------------------------------ #
        with open(CHECKPOINT_PATH / "config.json", "r") as f:
            config_dict = json.load(f)
            config_dict.pop("type", None)  # draccus does not expect this field

        config = draccus.decode(ACTConfig, config_dict)

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(CHECKPOINT_PATH / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        self.get_logger().info(
            f"RunACTLocal: loaded checkpoint from {CHECKPOINT_PATH} on {self.device}"
        )

        # ------------------------------------------------------------------ #
        # 2. Load normalisation statistics
        # ------------------------------------------------------------------ #
        stats = load_file(
            CHECKPOINT_PATH
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left": {
                "mean": get_stat("observation.images.left_camera.mean", (1, 3, 1, 1)),
                "std": get_stat("observation.images.left_camera.std", (1, 3, 1, 1)),
            },
            "center": {
                "mean": get_stat(
                    "observation.images.center_camera.mean", (1, 3, 1, 1)
                ),
                "std": get_stat("observation.images.center_camera.std", (1, 3, 1, 1)),
            },
            "right": {
                "mean": get_stat(
                    "observation.images.right_camera.mean", (1, 3, 1, 1)
                ),
                "std": get_stat("observation.images.right_camera.std", (1, 3, 1, 1)),
            },
        }

        # State: 26-dim; Action: 7-dim [x, y, z, qx, qy, qz, qw]
        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))

        self.image_scaling = 0.25  # must match recording scale

        self.get_logger().info("RunACTLocal: normalisation statistics loaded.")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _img_to_tensor(raw_img, device, scale, mean, std):
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    def _prepare_observations(self, obs_msg: Observation) -> dict:
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity
        state_np = np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs_msg.controller_state.tcp_error,
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )
        raw_state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw_state - self.state_mean) / self.state_std
        return obs

    @staticmethod
    def _clip_position_delta(current_pose: Pose, target: np.ndarray) -> np.ndarray:
        """Clip xyz delta so the arm cannot jump more than MAX_POSITION_DELTA per step."""
        delta = target[:3] - np.array(
            [
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z,
            ]
        )
        norm = np.linalg.norm(delta)
        if norm > MAX_POSITION_DELTA:
            target = target.copy()
            target[:3] = (
                np.array(
                    [
                        current_pose.position.x,
                        current_pose.position.y,
                        current_pose.position.z,
                    ]
                )
                + delta / norm * MAX_POSITION_DELTA
            )
        return target

    # ---------------------------------------------------------------------- #
    # Main loop
    # ---------------------------------------------------------------------- #

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(f"RunACTLocal.insert_cable() enter. Task: {task}")

        start_time = time.time()

        while time.time() - start_time < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warn("No observation received, skipping.")
                continue

            obs_tensors = self._prepare_observations(obs_msg)

            with torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)  # [1, 7]

            # Un-normalise: shape [7] → [x, y, z, qx, qy, qz, qw]
            action = (
                (normalized_action * self.action_std) + self.action_mean
            )[0].cpu().numpy()

            # Safety: limit how far the arm moves in a single step
            action = self._clip_position_delta(obs_msg.controller_state.tcp_pose, action)

            # Normalise quaternion to avoid controller warnings
            q = action[3:7]
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-6:
                q = q / q_norm

            pose = Pose(
                position=Point(x=float(action[0]), y=float(action[1]), z=float(action[2])),
                orientation=Quaternion(
                    x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])
                ),
            )

            self.get_logger().info(
                f"pos=({action[0]:.4f},{action[1]:.4f},{action[2]:.4f}) "
                f"q=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})"
            )

            # MODE_POSITION: send absolute target pose (base_link frame)
            self.set_pose_target(move_robot, pose, frame_id="base_link")
            send_feedback("in progress...")

            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        self.get_logger().info("RunACTLocal.insert_cable() done.")
        return True
