import numpy as np

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (PandaPickPlaceStudentEnv)

env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=True
)

grasp_policy = SAC.load(
    "models/checkpoints_random_grasp/"
    "sac_random_grasp_100000_steps"
)

transport_policy = SAC.load(
    "models/sac_random_transport_best"
)

num_episodes = 300

observations =[]
actions = []

successful_episodes = 0

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )

    terminated= False
    truncated = False

    use_transport_policy = False

    episode_observations = []
    episode_actions = []

    while not terminated and not truncated:

        if not use_transport_policy:

            teacher_observation = (
                observation[:28]
            )

            teacher_action, _ = (
                grasp_policy.predict(
                    teacher_observation,
                    deterministic=True
                )
            )


        else:

            teacher_observation = (
                observation[:34]
            )

            teacher_action , _ = (
                transport_policy.predict(
                    teacher_observation,
                    deterministic=True
                )
            )


        episode_observations.append(
            observation.copy()
        )

        episode_actions.append(teacher_action.copy())

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(teacher_action)

        if (
            not use_transport_policy and env.has_lifted
        ):
            use_transport_policy = True


    if info["is_success"]:

        successful_episodes += 1

        observations.extend(episode_observations)

        actions.extend(
            episode_actions
        )

observations = np.asarray(
    observations,
    dtype=np.float32
)

actions = np.asarray(
    actions,
    dtype=np.float32
)

np.savez(
    "models/"
    "hierarchical_random_cube_demos_37d.npz",

    observations = observations,
    actions=actions
)

print(
    "\n--- RANDOM-CUBE 37D DEMONSTRATION COLLECTION ---"
)

print(
    f"Successful episodes: "
    f"{successful_episodes}/{num_episodes}"
)

print(
    f"Total samples: "
    f"{len(observations)}"
)

print(
    f"Observation shape: "
    f"{observations.shape}"
)

print(
    f"Action shape: "
    f"{actions.shape}"
)

print(
    "\nSaved to:"
)

print(
    "models/"
    "hierarchical_random_cube_demos_37d.npz"
)


env.close()


