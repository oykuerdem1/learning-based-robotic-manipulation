from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.panda_reach_env import PandaReachEnv

env = PandaReachEnv(
    max_episode_steps=100,
    frame_skip=10,
    action_scale=0.05,
    success_threshold=0.05,
    randomize_target=True
)

env = Monitor(env)

model = SAC.load(
    "models/sac_panda_reach_fixed",
    env=env,
    device="cpu"
)

model.learning_starts= 0

checkpoint_callback = CheckpointCallback(
    save_freq= 25_000,
    save_path="./models/checkpoints_random_reach/",
    name_prefix="sac_random_reach",
    save_replay_buffer=True
)

model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback,
    log_interval=10,
    tb_log_name="SAC_random_reach",
    reset_num_timesteps=True
)

model.save("models/sac_panda_reach_random")

model.save_replay_buffer("models/sac_panda_reach_random_replay_buffer")

print("\nRandom-target training finished!")
print("Model saved to:")
print("models/sac_panda_reach_random.zip")

env.close()