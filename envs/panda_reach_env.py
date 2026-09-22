import numpy as np

import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco_menagerie as menagerie

class PandaReachEnv(gym.Env):
    def __init__(
            self,
            max_episode_steps = 100,
            frame_skip=10,
            action_scale = 0.05,
            success_threshold = 0.05,
            randomize_target=False
    ):
        super().__init__()

        self.model = menagerie.load("franka_emika_panda")
        self.data = mujoco.MjData(self.model)

        self.home_key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home"
        )

        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_threshold = success_threshold
        self.randomize_target = randomize_target

        self.step_count = 0

        self.fixed_target = np.array(
            [0.45, 0.10, 0.45],
            dtype=np.float32
        )

        self.target_position = self.fixed_target.copy()


        self.target_low = np.array(
            [0.40, -0.15, 0.42],
            dtype=np.float32
        )

        self.target_high = np.array(
            [0.55, 0.15, 0.58],
            dtype=np.float32
)

        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape= (7,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (20,),
            dtype= np.float32
        )

    def _get_observation(self):
        joint_positions = (
            self.data.qpos[:7].copy()
        )

        joint_velocities = (
            self.data.qvel[:7].copy()
        )

        ee_position = (
            self.data.body("hand").xpos.copy()
        )

        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            self.target_position
        ])

        return observation.astype(np.float32)

    def _get_distance(self):
        ee_position = (
            self.data.body("hand").xpos.copy()
        )

        distance = np.linalg.norm(ee_position - self.target_position)

        return float(distance)
    def _sample_target(self):

        target = self.np_random.uniform(
            low=self.target_low,
            high=self.target_high
        )

        return target.astype(np.float32)

    def reset(
            self,
            *,
            seed= None,
            options=None
    ):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            self.home_key_id
        )
        if self.randomize_target:
            self.target_position = (
                self._sample_target()
            )
        else:
            self.target_position = (
                self.fixed_target.copy()
    )

        self.data.ctrl[7] = 255

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.step_count = 0
        self.previous_lift_height = 0.0

        observation = (
            self._get_observation()
        )

        distance = (
            self._get_distance()
        )

        info = {
            "distance": distance,
            "is_success": (distance > self.success_threshold)
        }

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype = np.float32)

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        current_joint_positions = (
            self.data.qpos[:7].copy()
        )

        target_joint_positions = (
            current_joint_positions + self.action_scale * action
        )

        lower_limits = (
            self.model.actuator_ctrlrange[:7,0]
        )

        upper_limits = (
            self.model.actuator_ctrlrange[:7,1]
        )

        target_joint_positions = np.clip(
            target_joint_positions,
            lower_limits,
            upper_limits
        )

        self.data.ctrl[:7] = (
            target_joint_positions
        )

        self.data.ctrl[7] = 255

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        observation = (self._get_observation())

        distance = (self._get_distance())

        reward = -distance

        terminated = bool(
            distance < self.success_threshold
        )

        truncated = bool(
            self.step_count>= self.max_episode_steps
        )

        info = {
            "distance": distance,
            "is_success": terminated
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

