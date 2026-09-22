import numpy as np

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (PandaPickPlaceStudentEnv)

env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)

student = SAC.load(
    "models/sac_pick_place_bc_37d_balanced_best"
)

transport_teacher = SAC.load(
    "models/checkpoints_transport/"
    "sac_transport_75000_steps"
                             )

num_episodes = 200
observations = []
actions = []

successful_rollouts = 0
transport_samples = 0

for episode in range(num_episodes):

    observation, info = env.reset(
        seed=episode
    )


    terminated = False
    truncated = False

    episode_success = False

    while not terminated and not truncated:

        student_action, _ = student.predict(
            observation,
            deterministic=True
        )


        if not env.has_lifted:

            executed_action = student_action


        else:

            teacher_observation = observation[:34]

            teacher_action, _ = (
                transport_teacher.predict(
                    teacher_observation,
                    deterministic=True
                )
            )

            observations.append(
                observation.copy()
            )

            actions.append(
                teacher_action.copy()
            )

            transport_samples += 1

            if env.np_random.random() < 0.30:

                executed_action = teacher_action

            else:

                executed_action = student_action


        (
            observation,
            reward,
            terminated,
            truncated,
            info

        ) = env.step(executed_action)

    if info["is_success"]:

        successful_rollouts += 1


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
    "transport_dagger_37d_round1.npz",

    observations=observations,
    actions=actions
)

print(
    "\n--- TARGETED TRANSPORT DAGGER 37D ---"
)

print(
    f"Successful rollout episodes: "
    f"{successful_rollouts}/{num_episodes}"
)

print(
    f"Transport correction samples: "
    f"{transport_samples}"
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
    "transport_dagger_37d_round1.npz"
)


env.close()


            
        