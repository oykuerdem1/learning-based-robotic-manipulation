from stable_baselines3.common.env_checker import check_env
from envs.panda_reach_env import PandaReachEnv

env = PandaReachEnv()

print("Checking environment...")

check_env(
    env,
    warn = True
)

print("Environment check passed!")

observation, info = env.reset(
    seed = 42
)

print("\nObservation shape:")
print(observation.shape)

print("\nInitial distance:")
print(info["distance"])

print("\nRunning random actions...")

for step in range(10):

    action = env.action_space.sample()

    (
        observation, reward, terminated, truncated, info
    ) = env.step(action)

    print(
        f"step = {step:2d} | "
        f"reward = {reward:.3f} | "
        f"distance = {info['distance']:.3f}"
    )

    if terminated or truncated:

        observation, info = (
            env.reset()
        )

env.close()