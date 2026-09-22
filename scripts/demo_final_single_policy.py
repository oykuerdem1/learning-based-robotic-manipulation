import time

import mujoco.viewer

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (PandaPickPlaceStudentEnv)

env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=True
)


model = SAC.load(
    "models/"
    "sac_pick_place_random_cube_single_policy_final"
)

num_episodes = 5

observation, info = env.reset(
    seed=42
)

with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:


    for episode in range(num_episodes):

        observation, info = env.reset(
            seed=episode
        )

        terminated =False
        truncated = False

        step = 0

        print(
            f"\nEpisode {episode + 1}"
        )

        while (not terminated and not truncated and viewer.is_running()):

            action,_ = model.predict(
                observation,
                deterministic=True
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            step += 1

            viewer.sync()

            time.sleep(0.01)
        print(
            "Success:",
            info["is_success"]
        )

        print(
            "Steps:",
            step
        )

        print(
            "Final goal distance:",
            round(
                info["goal_xy_distance"],
                3
            ),
            "m"
        )

        print(
            "Final z error:",
            round(
                info["goal_z_error"],
                3
            ),
            "m"
        )

        if viewer.is_running():

            time.sleep(1.0)

env.close()


        
