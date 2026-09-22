import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)

from stable_baselines3 import SAC

from envs.panda_pick_place_student_env import (PandaPickPlaceStudentEnv)


np.random.seed(42)
torch.manual_seed(42)


original_data = np.load(
    "models/hierarchical_pick_place_demos_37d.npz"
)

original_observations = (
    original_data["observations"].astype(np.float32)
)

original_actions = (
    original_data["actions"].astype(np.float32)
)


dagger_data = np.load(
    "models/transport_dagger_37d_round1.npz"
)

dagger_observations = (
    dagger_data["observations"].astype(np.float32)
)

dagger_actions = (
    dagger_data["actions"].astype(np.float32)
)

print(
    "Original samples:",
    len(original_observations)
)

print(
    "DAgger samples:",
    len(dagger_observations)
)


observations = np.concatenate(
    [
        original_observations,
        dagger_observations
    ],
    axis=0
)

actions = np.concatenate(
    [
        original_actions,
        dagger_actions
    ],
    axis=0
)

source = np.concatenate(
    [
        np.zeros(
            len(original_observations),
            dtype=np.int32
        ),

        np.ones(
            len(dagger_observations),
            dtype=np.int32
        )
    ]
)

print(
    "Combined observations:",
    observations.shape
)

print(
    "Combined actions:",
    actions.shape
)

num_samples = len(observations)

indices = np.random.permutation(
    num_samples
)

train_size = int(
    0.9 * num_samples
)

train_indices = indices[:train_size]

val_indices = indices[train_size:]

train_dataset = TensorDataset(
    torch.from_numpy(
        observations[train_indices]
    ),
    torch.from_numpy(
        actions[train_indices]
    )
)

val_dataset = TensorDataset(
    torch.from_numpy(
        observations[val_indices]
    ),

    torch.from_numpy(
        actions[val_indices]
    )
)


train_observations = observations[train_indices]

train_source = source[train_indices]

left_contact = (
    train_observations[:,34] > 0.5
)

right_contact = (
    train_observations[:,35] > 0.5
)

lift_height = (
    train_observations[:, 36]
)

both_contacts = (
    left_contact & right_contact
)

grasped_low = (
    both_contacts
    & (lift_height < 0.01)
)

lifting = (
    both_contacts & (lift_height >= 0.01)
    & (lift_height < 0.04)
)

sample_weights = np.ones(
    len(train_indices),
    dtype=np.float32
)

sample_weights[train_source==1] = 2.0

sample_weights[grasped_low] = np.maximum(
    sample_weights[grasped_low],4.0
)

sample_weights[lifting] = np.maximum(
    sample_weights[lifting],
    4.0
)

sampler = WeightedRandomSampler(
    weights = torch.from_numpy(
        sample_weights
    ),

    num_samples = len(sample_weights),

    replacement =True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False
)


env = PandaPickPlaceStudentEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)


model = SAC.load(
    "models/sac_pick_place_bc_37d_balanced_best",
    env=env,
    device="cpu"
)

actor = model.actor

optimizer = actor.optimizer

for param_group in optimizer.param_groups:

    param_group["lr"] = 5e-5


loss_function = nn.MSELoss()

num_epochs = 15

best_val_loss =np.inf

for epoch in range(1,num_epochs+1):

    actor.train()

    train_loss_sum = 0.0
    train_samples =0

    for (
        obs_batch,
        action_batch
    ) in train_loader:

        obs_batch = obs_batch.to(
            model.device
        )

        action_batch = action_batch.to(
            model.device
        )

        predicted_actions = actor(
            obs_batch,
            deterministic=True
        )

        loss = loss_function(
            predicted_actions,
            action_batch
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = (
            obs_batch.shape[0]
        )

        train_loss_sum += (
            loss.item() * batch_size
        )

        train_samples += (batch_size)

    train_loss = (
        train_loss_sum / train_samples
    )

    actor.eval()

    val_loss_sum= 0.0

    val_samples = 0

    with torch.no_grad():

        for (
            obs_batch,
            action_batch
        ) in val_loader:

            obs_batch = obs_batch.to(model.device)

            action_batch = action_batch.to(model.device)

            predicted_actions = actor(
                obs_batch,
                deterministic=True
            )

            loss = loss_function(
                predicted_actions,
                action_batch
            )

            batch_size = (
                obs_batch.shape[0]
            )

            val_loss_sum += (
                loss.item() * batch_size
            )

            val_samples += (batch_size)

    val_loss = (
        val_loss_sum / val_samples
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model.save(
            "models/"
            "sac_pick_place_dagger_37d_round1_best"
        )

        if epoch in [5,10,15]:

            model.save(
                "models/"
                f"sac_pick_place_dagger_37d_round1_epoch_{epoch}"
            )


        if (
            epoch == 1 or epoch % 5== 0
        ):

            print(
            f"Epoch {epoch:2d}/{num_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

model.save(
    "models/"
    "sac_pick_place_dagger_37d_round1_final"
)

print(
    "\nTargeted transport DAgger refinement finished!"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "\nSaved rollout checkpoints:"
)

print(
    "epoch 5"
)

print(
    "epoch 10"
)

print(
    "epoch 15"
)


env.close()




