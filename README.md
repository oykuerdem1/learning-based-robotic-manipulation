# Learning Based Robotic Manipulation in Mujoco

A robotic manipulation project using a Franka Panda robot in Mujoco using reinforcement learning.

The final policy : reach, grasp, lift, transport, lower, release

## Results:
Fixed cube success: 100%
Random cube success: 89.6% (tested on 500 episodes)

## Project:
- Used a SAC policy for reaching and grasping
- Modified reward systems according to the behaviour of the robot during different stages of the task
- Applied curriculum learning by training transporting stage seperately from the stages that succeded in lifting and grasping.
- Applied DAgger so the student could learn from teacher correction on with its own interactions
- Generalized training with random cube positions after it succeeded fixed cube position
- Finally, tested the final policy on 500 different random cube positions and achieved 89.6% success


## Final Trained Model

models/sac_pick_place_random_cube_single_policy_final.zip

## Running the project

source .venv/bin/activate
pip install -r requirements.txt
mjpython -m scripts.demo_final_single_policy

## Evaluating the project

python -m scripts.evaluate_final_random_cube
python -m scripts.evaluate_final_fixed_cube_

