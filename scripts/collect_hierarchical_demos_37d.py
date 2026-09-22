import numpy as np

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (
    PandaPickPlaceStudentEnv
)

env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)

grasp_policy = SAC.load(
    "models/checkpoints_pick_place_v1/"
    "sac_pick_place_v1_100000_steps"
)

transport_policy = SAC.load(
    "models/checkpoints_transport/"
    "sac_transport_75000_steps"
)


num_episodes = 200

observations = []
actions = []

successful_episodes = 0

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )

    terminated = False
    truncated =False

    use_transport_policy = False
    episode_observations = []
    episode_actions = []

    while not terminated and not truncated:

        teacher_observation = (
            observation[:34]
        )

        if not use_transport_policy:

            teacher_action,_ = (
                grasp_policy.predict(
                    teacher_observation,
                    deterministic=True
                )
            )

        else:

            teacher_action,_ = transport_policy.predict(
                teacher_observation, deterministic=True
            )


        episode_observations.append(
            observation.copy()
        )

        episode_actions.append(
            teacher_action.copy()
        )

        noise = env.np_random.normal(
            loc=0.0,
            scale=0.02,
            size=teacher_action.shape
        )

        executed_action = np.clip(
            teacher_action + noise,
            -1.0,
            1.0
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(executed_action)


        if (
            not use_transport_policy
            and env.has_lifted
        ):
            use_transport_policy = True


    if info["is_success"]:

        successful_episodes+=1

        observations.extend(
            episode_observations
        )

        actions.extend(episode_actions)

observations = np.asarray(
    observations,
    dtype = np.float32
)

actions = np.asarray(
    actions,
    dtype=np.float32
)


np.savez(
    "models/"
    "hierarchical_pick_place_demos_37d.npz",
    observations=observations,
    actions=actions
)

print(
    "\n--- 37D HIERARCHICAL DEMONSTRATION COLLECTION ---"
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
    "hierarchical_pick_place_demos_37d.npz"
)


env.close()