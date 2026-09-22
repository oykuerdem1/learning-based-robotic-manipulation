# Learning-Based Robotic Manipulation in MuJoCo

A robotic manipulation project using a Franka Panda robot in MuJoCo with reinforcement learning.

The final policy performs:

**Reach → Grasp → Lift → Transport → Lower → Release**

## Results

**Fixed cube success:** 100%  
**Random cube success:** 89.6% (tested on 500 episodes)

## Project

- Used a SAC policy for reaching and grasping.

- Modified reward functions according to the behavior of the robot during different stages of the task.

- Applied curriculum learning by training the transport stage separately after successfully learning grasping and lifting.

- Applied DAgger so the student policy could learn from teacher corrections during its own interactions with the environment.

- Generalized the policy by training with randomized cube positions after successfully solving the fixed-cube task.

- Finally, tested the final policy on 500 randomized episodes and achieved an 89.6% success rate.

## Final Trained Model

`models/sac_pick_place_random_cube_single_policy_final.zip`

## Running the Project

```bash
source .venv/bin/activate
pip install -r requirements.txt
mjpython -m scripts.demo_final_single_policy


## Evaluating the project

python -m scripts.evaluate_final_random_cube
python -m scripts.evaluate_final_fixed_cube


