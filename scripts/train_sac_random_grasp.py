from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.panda_grasp_env import PandaGraspEnv

env = PandaGraspEnv(
    max_episode_steps=150,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=True
)

env = Monitor(
    env,
    info_keywords=("is_success",)
)

model = SAC.load(
    "models/checkpoints_grasp_v3/"
    "sac_grasp_v3_100000_steps",
    env = env,
    device= "cpu"
)

model.learning_starts = 0

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path = "./models/checkpoints_random_grasp/",
    name_prefix = "sac_random_grasp",
    save_replay_buffer=True
)

model.learn(
    total_timesteps=300_000,
    callback=checkpoint_callback,
    log_interval=10,
    tb_log_name="SAC_random_grasp",
    reset_num_timesteps=True
)

model.save(
    "models/sac_panda_grasp_random"
)

model.save_replay_buffer(
    "models/sac_panda_grasp_random_replay_buffer"
)

print("\nRandom-cube grasp training finished!")

print(
    "Model saved to "
    "models/sac_panda_grasp_random.zip"
)

env.close()