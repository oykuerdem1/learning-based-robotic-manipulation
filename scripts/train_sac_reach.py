from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from envs.panda_reach_env import PandaReachEnv

env = PandaReachEnv(
    max_episode_steps=100,
    frame_skip=10,
    action_scale=0.05,
    success_threshold=0.05
)
env = Monitor(env)

model = SAC(
    policy="MlpPolicy",
    env = env,

    learning_rate=3e-4,

    buffer_size=100_000,
    learning_starts=1_000,

    batch_size=256,

    tau=0.005,
    gamma=0.99,

    train_freq=1,
    gradient_steps=1,

    ent_coef="auto",

    verbose=1,

    tensorboard_log="./logs/sac_reach/",
    seed=42,
    device = "cpu"
)

model.learn(total_timesteps=200_000,
            log_interval=10,
            tb_log_name="SAC_fixed_v2")

model.save(
    "models/sac_panda_reach_fixed_v2"
)
model.save_replay_buffer(
    "models/sac_panda_reach_fixed_replay_buffer"
)

print("\nTraining finished!")
print("Model saved to:")
print("models/sac_panda_reach_fixed.zip")


env.close()