import numpy as np

from stable_baselines3 import SAC

from envs.panda_pick_place_env import PandaPickPlaceEnv

env = PandaPickPlaceEnv(
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

num_episodes = 100

successes = 0
switch_count = 0

episode_lengths =[]
switch_steps = []
final_goal_distances = []

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )

    terminated = False
    truncated = False

    step = 0
    use_transport_policy =False
    switch_step = None

    while not terminated and not truncated:

        if not use_transport_policy:

            action,_ = grasp_policy.predict(
                observation,
                deterministic=True
            )

        else:
            action,_ = transport_policy.predict(
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

        if (
            not use_transport_policy
            and env.has_lifted
        ):

            use_transport_policy = True

            switch_step = step

            switch_count += 1

    successes += int(info["is_success"])

    episode_lengths.append(
        step
    )

    final_goal_distances.append(info["goal_xy_distance"])

    if switch_step is not None:
        switch_steps.append(switch_step)

print(
    "\n--- HIERARCHICAL FULL PICK-AND-PLACE ---"
)

print(
    f"Switch-to-transport rate: "
    f"{100 * switch_count / num_episodes:.1f}%"
)

print(
    f"Success rate: "
    f"{100 * successes / num_episodes:.1f}%"
)

print(
    f"Mean episode length: "
    f"{np.mean(episode_lengths):.1f} steps"
)

if switch_steps:

    print(
        f"Mean grasp/switch step: "
        f"{np.mean(switch_steps):.1f}"
    )

print(
    f"Mean final goal distance: "
    f"{np.mean(final_goal_distances):.3f} m"
)


env.close()


