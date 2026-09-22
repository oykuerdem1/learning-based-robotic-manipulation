from stable_baselines3 import SAC

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.panda_pick_place_env import PandaPickPlaceEnv

env = PandaPickPlaceEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)

env = Monitor(
    env,
    info_keywords=("is_success",)
)

model = SAC(
    "MlpPolicy",
    env,

    learning_rate=3e-4,

    buffer_size=500_000,
    learning_starts=10_000,

    batch_size=256,

    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,

    ent_coef="auto",
    verbose=1,

    tensorboard_log="./logs/sac_pick_place/",

    seed=42,
    device="cpu"
)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,

    save_path="./models/checkpoints_pick_place_v1/",

    name_prefix="sac_pick_place_v1",
    save_replay_buffer=True
)

model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,

    log_interval=10,

    tb_log_name="SAC_pick_place_fixed_v1"
)

model.save("models/sac_panda_pick_place_fixed_v1")

model.save_replay_buffer(
    "models/sac_panda_pick_place_fixed_v1_replay_buffer"
)

print(
    "\nFixed pick-and-place training finished!"
)

env.close()

