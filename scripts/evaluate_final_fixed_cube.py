import numpy as np

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (PandaPickPlaceStudentEnv)

env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,

    randomize_cube=False
)

model = SAC.load("models/sac_pick_place_random_cube_single_policy_final")

num_episodes = 100

successes = 0

grasp_count =0
lift_count = 0
transport_count = 0
lower_count = 0
release_count = 0

episode_lengths = []

max_lift_heights = []
min_goal_distances = []
final_goal_distances = []

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )

    terminated = False
    truncated = False

    step = 0

    ever_grasped = False
    ever_lifted = False
    ever_transported = False
    ever_lowered = False
    ever_released = False

    max_lift = 0.0
    min_goal_distance = np.inf

    while not terminated and not truncated:

        action, _ = model.predict(
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


        both_contacts = (
            info["left_contact"] and info["right_contact"]
        )

        if both_contacts:
            ever_grasped = True

        if env.has_lifted:
            ever_lifted = True

        if (
            ever_lifted and info["goal_xy_distance"] < 0.06
        ):

            ever_transported = True

        if (
            ever_transported and info["goal_z_error"] < 0.02
        ):
            ever_lowered = True

        if (
            ever_lowered 
            and info["released"]
        ):
            ever_released = True

        max_lift = max(
            max_lift,
            info["lift_height"]
        )

        min_goal_distance = min(
            min_goal_distance,
            info["goal_xy_distance"]
        )

    successes += int(info["is_success"])

    grasp_count += int(ever_grasped)
    lift_count += int(ever_lifted)
    transport_count += int(ever_transported)
    lower_count += int(ever_lowered)
    release_count += int(ever_released)

    episode_lengths.append(step)

    max_lift_heights.append(max_lift)

    min_goal_distances.append(min_goal_distance)

    final_goal_distances.append(info["goal_xy_distance"])

print(
    "\n--- FINAL SINGLE POLICY: FIXED CUBE REGRESSION ---"
)

print(
    f"Grasp rate: "
    f"{100 * grasp_count / num_episodes:.1f}%"
)

print(
    f"Lift rate: "
    f"{100 * lift_count / num_episodes:.1f}%"
)

print(
    f"Transport rate: "
    f"{100 * transport_count / num_episodes:.1f}%"
)

print(
    f"Lower rate: "
    f"{100 * lower_count / num_episodes:.1f}%"
)

print(
    f"Release rate: "
    f"{100 * release_count / num_episodes:.1f}%"
)

print(
    f"Success rate: "
    f"{100 * successes / num_episodes:.1f}%"
)


print(
    f"\nMean episode length: "
    f"{np.mean(episode_lengths):.1f}"
)

print(
    f"Mean max lift height: "
    f"{np.mean(max_lift_heights):.3f} m"
)

print(
    f"Mean minimum goal distance: "
    f"{np.mean(min_goal_distances):.3f} m"
)

print(
    f"Mean final goal distance: "
    f"{np.mean(final_goal_distances):.3f} m"
)


env.close()