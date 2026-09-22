from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import (CheckpointCallback)

from stable_baselines3.common.utils import (get_schedule_fn)

from envs.panda_random_transport_env import (PandaRandomTransportEnv)

env = PandaRandomTransportEnv(
    max_episode_steps=150,
    frame_skip=10,
    action_scale=0.04
)

model = SAC.load(
    "models/checkpoints_transport/"
    "sac_transport_75000_steps",
    env=env,
    device="cpu"
)


model.learning_starts=0

new_learning_rate = 1e-4

model.learning_rate = new_learning_rate

model.lr_schedule = get_schedule_fn(new_learning_rate)


for param_group in (model.actor.optimizer.param_groups):
    param_group["lr"] = new_learning_rate

for param_group in (model.critic.optimizer.param_groups):
    param_group["lr"] = new_learning_rate


if model.ent_coef_optimizer is not None:

    for param_group in (model.ent_coef_optimizer.param_groups):
        param_group["lr"] = new_learning_rate


checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path=(
        "models/"
        "checkpoints_random_transport/"
    ),

    name_prefix= ("sac_random_transport")
)

model.learn(
    total_timesteps=100_000,
    callback=checkpoint_callback,
    reset_num_timesteps=True
)

model.save(
    "models/"
    "sac_random_transport_final"
)

print(
    "\nRandom transport fine-tuning finished!"
)

print(
    "Checkpoints saved at:"
)

print(
    "25k, 50k, 75k, 100k steps"
)


env.close()