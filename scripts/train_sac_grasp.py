from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.panda_grasp_env import PandaGraspEnv

env = PandaGraspEnv(
    max_episode_steps=150,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)

env = Monitor(
    env,
    info_keywords=("is_success",)
)

model = SAC(
    policy = "MlpPolicy",
    env = env,

    learning_rate=3e-4,

    buffer_size = 200_000,
    learning_starts=5_000,

    batch_size=256,

    tau = 0.005,
    gamma = 0.99,

    train_freq=1,
    gradient_steps=1,

    ent_coef="auto",

    verbose=1,

    tensorboard_log="./logs/sac_grasp",

    seed = 42,
    device = "cpu"
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./models/checkpoints_grasp_v3/",
    name_prefix="sac_grasp_v3",
    save_replay_buffer=True
)

model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,
    log_interval=10,
    tb_log_name="SAC_grasp_fixed_v3"
)

model.save(
    "models/sac_panda_grasp_fixed_v3"
)

model.save_replay_buffer(
    "models/sac_panda_grasp_fixed_v3_replay_buffer"
)

print("\nGrasp v3 training finished!")
print(
    "Model saved to "
    "models/sac_panda_grasp_fixed_v3.zip"
)

env.close()