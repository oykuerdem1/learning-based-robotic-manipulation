import numpy as np
import mujoco

from envs.panda_pick_place_env import (PandaPickPlaceEnv)

class PandaRandomTransportEnv(
    PandaPickPlaceEnv
):
    def __init__(
            self,
            state_path = (
                "models/"
                "random_cube_lifted_states.npz"
            ),
            max_episode_steps = 150,
            frame_skip= 10,
            action_scale =0.04
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_scale=action_scale,
            randomize_cube=False
        )

        data = np.load(
            state_path
        )

        self.saved_qpos = data["qpos"]

        self.saved_ctrl = data["ctrl"]

        self.num_saved_states = len(self.saved_qpos)

        self.current_state_index = None

    def reset(
            self,
            *,
            seed=None,
            options= None
    ):

        observation, info = super().reset(
            seed=seed,
            options=options
        )

        state_index = int(
            self.np_random.integers(
                0, self.num_saved_states
            )
        )

        self.current_state_index = (
            state_index
        )

        self.data.qpos[:] = (
            self.saved_qpos[state_index]

        )

        self.data.qvel[:] = 0.0

        self.data.ctrl[:] = (
            self.saved_ctrl[state_index]
        )

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0

        self.has_lifted = True
        self.has_grasped = True

        cube_position = (
            self.data.body(
                "cube"
            ).xpos.copy()
        )

        lift_height = max(
            0.0,
            cube_position[2] - self.initial_cube_z
        )

        self.best_lift_height = min(
            lift_height, 0.05
        )

        goal_xy_distance = np.linalg.norm(
            cube_position[:2] - self.goal_position[:2]
        )

        self.best_goal_xy_distance =(goal_xy_distance)

        self.previous_goal_xy_distance = (goal_xy_distance)

        self.best_place_z_error = np.inf

        self.reached_goal_region = (
            goal_xy_distance < 0.06
        )

        observation = (
            self._get_observation()
        )

        return observation, info