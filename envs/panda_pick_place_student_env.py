import numpy as np
from gymnasium import spaces

from envs.panda_pick_place_env import PandaPickPlaceEnv

class PandaPickPlaceStudentEnv(PandaPickPlaceEnv):

    def __init__(
            self,
            max_episode_steps = 200,
            frame_skip = 10,
            action_scale = 0.04,
            randomize_cube = False
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            action_scale=action_scale,
            randomize_cube=randomize_cube
        )


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(37,),
            dtype=np.float32
        )

    def _get_observation(self):

        base_observation = (
            super()._get_observation()
        )

        left_contact = (
            self._bodies_in_contact(
                self.cube_body_id,
                self.left_finger_body_id
            )
        )

        right_contact = (
            self._bodies_in_contact(
                self.cube_body_id,
                self.right_finger_body_id
            )
        )

        cube_position = (
            self.data.body("cube").xpos.copy()
        )

        lift_height = max(
            0.0,
            cube_position[2] - self.initial_cube_z
        )

        extra_observation = np.array(
            [
                float(left_contact),
                float(right_contact),
                float(lift_height)
            ],
            dtype= np.float32
        )

        observation = np.concatenate(
            [
                base_observation,
                extra_observation
            ]
        )

        return observation.astype(np.float32)