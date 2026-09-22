import numpy as np
import mujoco

from envs.panda_pick_place_env import PandaPickPlaceEnv

class PandaTransportEnv(PandaPickPlaceEnv):

    def __init__(
            self, 
            max_episode_steps=150, 
            frame_skip=10, 
            action_scale=0.04, 
            state_path = "models/pick_place_lifted_start.npz"):

        super().__init__(
            max_episode_steps, 
            frame_skip, 
            action_scale, 
            randomize_cube = False)

        state = np.load(state_path)

        self.lifted_qpos = (
            state["qpos"].copy()
        )

        self.lifted_ctrl = (
            state["ctrl"].copy()
        )

    def reset(
            self,
            *,
            seed = None,
            options = None
    ):

        super().reset(
            seed=seed,
            options=options
                            )


        self.data.qpos[:] = (
            self.lifted_qpos
        )


        self.data.qvel[:] = 0.0

        self.data.ctrl[:] = (
            self.lifted_ctrl
        )

        mujoco.mj_forward(
            self.model,
            self.data
        )

        self.step_count = 0

        self.initial_cube_z = (
            self.fixed_cube_position[2]
        )

        cube_position = (
            self.data.body("cube").xpos.copy()
        )

        self.has_grasped = True
        self.has_lifted = True

        self.best_goal_xy_distance = (
            np.linalg.norm(
                cube_position[:2] - self.goal_position[:2]
            )
        )

        self.previous_goal_xy_distance = (
            self.best_goal_xy_distance
        )

        self.best_place_z_error = np.inf

        self.reached_goal_region = False

        observation = (
              self._get_observation()
        )

        info = (
            self._get_task_info()
        )

        return observation, info

"""
normal PickPlaceEnv reset
        ↓
saved state'i yükle
        ↓
robot + cube havadaki grasp pozisyonuna işinlaniyor
        ↓
has_grasped = True
has_lifted  = True
        ↓
RL yalnizca kalan görevi öğreniyor
"""
      