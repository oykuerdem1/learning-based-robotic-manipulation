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

num_episodes = 300

qpos_states = []
ctrl_states = []

cube_positions = []
lift_heights = []

successful_lifts = 0

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )

    terminated = False
    truncated = False

    while not terminated and not truncated:

        grasp_observation = observation[:28]

        action,_ = grasp_policy.predict(
            grasp_observation,
            deterministic=True
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        if env.has_lifted:

            qpos_states.append(
                env.data.qpos.copy()
            )

            ctrl_states.append(
                env.data.ctrl.copy()
            )

            cube_positions.append(env.data.body(
                "cube"
            ).xpos.copy()
            )

            lift_heights.append(info["lift_height"])

            successful_lifts += 1

            break
qpos_states = np.asarray(
    qpos_states,
    dtype=np.float64
)

ctrl_states = np.asarray(
    ctrl_states,
    dtype=np.float64
)

cube_positions = np.asarray(
    cube_positions,
    dtype=np.float64
)

lift_heights = np.asarray(
    lift_heights,
    dtype=np.float64
)

np.savez(
    "models/random_cube_lifted_states.npz",

    qpos = qpos_states,
    ctrl = ctrl_states,
    cube_positions = cube_positions,
    lift_heights=lift_heights
)

print(
    "\n--- RANDOM CUBE LIFTED STATE COLLECTION ---"
)

print(
    f"Successful lifts: "
    f"{successful_lifts}/{num_episodes}"
)

print(
    f"Saved states: "
    f"{len(qpos_states)}"
)

print(
    "qpos shape:",
    qpos_states.shape
)

print(
    "ctrl shape:",
    ctrl_states.shape
)

if len(lift_heights) > 0:

    print(
        f"Mean lift height: "
        f"{np.mean(lift_heights):.3f} m"
    )

    print(
        "Cube x range:",
        f"{cube_positions[:, 0].min():.3f}",
        "to",
        f"{cube_positions[:, 0].max():.3f}"
    )

    print(
        "Cube y range:",
        f"{cube_positions[:, 1].min():.3f}",
        "to",
        f"{cube_positions[:, 1].max():.3f}"
    )


print(
    "\nSaved to:"
)

print(
    "models/random_cube_lifted_states.npz"
)


env.close()