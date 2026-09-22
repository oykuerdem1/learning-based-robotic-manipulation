from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.panda_transport_env import PandaTransportEnv

env = PandaTransportEnv(
    max_episode_steps=150,
    frame_skip=10,
    action_scale=0.04
)

env = Monitor(
    env, info_keywords=("is_success",)
)

model = SAC(
    "MlpPolicy",
    env,

    learning_rate=3e-4,

    buffer_size=300_000,
    learning_starts=5_000,

    batch_size=256,

    tau=0.005,
    gamma=0.99,

    train_freq=1,
    gradient_steps=1,

    ent_coef="auto",

    verbose=1,

    tensorboard_log="./logs/sac_transport/",

    seed = 42,
    device = "cpu"
)

checkpoint_callback = CheckpointCallback(

    save_freq=25_000,

    save_path="./models/checkpoints_transport/",

    name_prefix="sac_transport",

    save_replay_buffer=True
)

model.learn(
    total_timesteps=250_000,

    callback=checkpoint_callback,

    log_interval=10,
    tb_log_name="SAC_transport_curriculum"
)

model.save(
    "models/sac_panda_transport"
)

model.save_replay_buffer(
    "models/sac_panda_transport_replay_buffer"
)

print(
    "\nTransport curriculum training finished!"
)

env.close()
