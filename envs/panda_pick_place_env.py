import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco

from envs.panda_grasp_scene import create_grasp_scene

class PandaPickPlaceEnv(gym.Env):

    def __init__(
            self,
            max_episode_steps = 150,
            frame_skip=10,
            action_scale = 0.04,
            randomize_cube = False
    ):
        super().__init__()

        self.model, self.data = create_grasp_scene(
            add_place_target=True
        )

        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip

        self.action_scale = action_scale
        self.randomize_cube = randomize_cube

        self.step_count = 0


        self.home_arm = np.array([
            0.0,
            0.0,
            0.0,
            -1.57079,
            0.0,
            1.57079,
            -0.7853
        ])

        self.arm_qpos_ids = np.array([
            self.model.joint(f"joint{i}").qposadr[0]
            for i in range(1,8)
        ])

        self.arm_dof_ids = np.array([
            self.model.joint(
                f"joint{i}"
            ).dofadr[0]
            for i in range(1,8)
        ])


        self.finger_qpos_ids = np.array([
            self.model.joint("finger_joint1").qposadr[0],

            self.model.joint(
                "finger_joint2"
            ).qposadr[0]
        ])


        cube_body_id = self.model.body("cube").id

        cube_joint_id = (
            self.model.body("cube").jntadr[0]
        )

        self.cube_qpos_adr = (
            self.model.jnt_qposadr[cube_joint_id]
        )

        self.cube_dof_adr = (
            self.model.jnt_dofadr[cube_joint_id]
        )

        self.previous_goal_xy_distance = np.inf
        self.reached_goal_region = False


        self.cube_body_id = cube_body_id

        self.goal_position = np.array(
            [0.50, 0.20, 0.425],
            dtype=np.float64
        )
        self.best_lift_height = 0.0
        self.best_goal_xy_distance = np.inf
        self.best_place_z_error = np.inf
        self.has_grasped = False
        self.has_lifted = False

        self.left_finger_body_id = (
            self.model.body("left_finger").id
        )

        self.right_finger_body_id = (
            self.model.body("right_finger").id
        )


        self.lower_limits = (
            self.model.actuator_ctrlrange[:7,0]
        )

        self.upper_limits = (
            self.model.actuator_ctrlrange[:7,1]
        )

        self.fixed_cube_position = np.array([
            0.50,
            0.00,
            0.425
        ])

        self.cube_xy_low = np.array([
            0.44,
            -0.08
        ])

        self.cube_xy_high = np.array([
            0.54,
            0.08
        ])

        self.initial_cube_z = (
            self.fixed_cube_position[2]
        )

        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (8,),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = -np.inf,
            high= np.inf,
            shape =(34,),
            dtype = np.float32
        )

    def _bodies_in_contact(
            self,
            body_a,
            body_b
    ):
        for i in range(self.data.ncon):
            contact = (
                self.data.contact[i]
            )

            body1 = (
                self.model.geom_bodyid[contact.geom1]
            )

            body2 = (
                self.model.geom_bodyid[contact.geom2]
            )

            if (
                body1 == body_a and body2 == body_b
            ) or (body1 == body_b and body_a == body2):
                return True

        return False

    def _get_observation(self):

        arm_qpos = (
            self.data.qpos[self.arm_qpos_ids].copy()
        )

        arm_qvel = (
            self.data.qvel[self.arm_dof_ids].copy()
        )

        grasp_position = (
            self.data.site("grasp_site").xpos.copy()
        )

        cube_position = (
            self.data.body("cube").xpos.copy()
        )

        relative_position = (cube_position - grasp_position)

        cube_to_goal = (
            self.goal_position - cube_position
        )

        finger_positions = (
            self.data.qpos[self.finger_qpos_ids].copy()
        )

        cube_linear_velocity = (
            self.data.qvel[self.cube_dof_adr:self.cube_dof_adr+3].copy()
        )

        observation = np.concatenate([
            arm_qpos,
            arm_qvel,
            grasp_position,
            cube_position,
            relative_position,
            finger_positions,
            cube_linear_velocity,
            self.goal_position,
            cube_to_goal
        ])

        return observation.astype(np.float32)

    def _get_task_info(self):

        grasp_position = (
            self.data.site("grasp_site").xpos.copy()
        )

        cube_position = (
            self.data.body(
                "cube"
            ).xpos.copy()
        )


        distance = np.linalg.norm(grasp_position - cube_position)

        left_contact =(self._bodies_in_contact(self.cube_body_id,self.left_finger_body_id))

        right_contact = (
            self._bodies_in_contact(
                self.cube_body_id,
                self.right_finger_body_id
            )
        )

        gripper_command_open = (
            self.data.ctrl[7] > 200.0
        )


        lift_height = max(
            0.0,
            cube_position[2] - self.initial_cube_z
        )

        goal_xy_distance = np.linalg.norm(
            cube_position[:2] - self.goal_position[:2]
        )

        goal_z_error = abs(
            cube_position[2] - self.goal_position[2]
        )

        released = (
            gripper_command_open and not (
                left_contact and right_contact
            )
        )

        is_success = (
            self.has_lifted
            and goal_xy_distance < 0.05
            and goal_z_error < 0.02
            and released
        )



        return {
            "distance": float(distance),
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "lift_height": float(lift_height),
            "goal_xy_distance": float(goal_xy_distance),
            "goal_z_error": float(goal_z_error),
            "released": bool(released),
            "is_success": bool(is_success)
        }


    def _compute_reward(self, info):

        reach_reward = -info["distance"]

        both_contacts = (
            info["left_contact"] and info["right_contact"])

        grasp_reward = 0.0

        if (
                both_contacts and not self.has_grasped
        ):
            grasp_reward = 1.0
            self.has_grasped = True

        capped_lift = min(info["lift_height"],0.05)

        lift_reward = 0.0

        if both_contacts:
            lift_progress = max(
                0.0,
                capped_lift - self.best_lift_height
            )

            lift_reward = (
                100.0 * lift_progress
            )

            self.best_lift_height = max(
                self.best_lift_height, capped_lift
            )

        if (
            self.has_grasped and info["lift_height"] >= 0.04
        ):
            self.has_lifted = True

        transport_reward = 0.0
        goal_region_bonus = 0.0

        if self.has_lifted:
            goal_progress = (
                self.previous_goal_xy_distance - info["goal_xy_distance"]
            )

            if both_contacts:

                transport_reward = (
                    150.0 * goal_progress
                )

            self.previous_goal_xy_distance = (
                info["goal_xy_distance"]
            )

            if (
                both_contacts and info["goal_xy_distance"] < 0.06
                and not self.reached_goal_region
            ):
                goal_region_bonus = 5.0

                self.reached_goal_region = True


        place_reward = 0.0

        if (
            self.has_lifted and info["goal_xy_distance"] < 0.06
        ):
            if np.isinf(self.best_place_z_error):
                self.best_place_z_error = (
                    info["goal_z_error"]
                )

            else:
                place_progress = max(
                    0.0,
                    self.best_place_z_error - info["goal_z_error"]
                )

                place_reward = (
                    50.0 * place_progress
                )

                self.best_place_z_error = min(
                    self.best_place_z_error,
                    info["goal_z_error"]
                )

        success_bonus = (
            25.0 if info["is_success"] else 0.0
        )

        reward = (
            reach_reward 
            + grasp_reward 
            + lift_reward 
            + transport_reward 
            + place_reward 
            + success_bonus 
            + goal_region_bonus
        )

        return float(reward)
    



    def reset(
            self,
            *,
            seed=None,
            options = None
    ):
        super().reset(seed = seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.data.qpos[self.arm_qpos_ids] = self.home_arm

        self.data.qpos[self.finger_qpos_ids] = 0.04

        cube_position = (
            self.fixed_cube_position.copy()
        )

        if self.randomize_cube:

            cube_position[:2] =(
                self.np_random.uniform(
                    low= self.cube_xy_low,
                    high=self.cube_xy_high
                )
            )

        cube_q = (
            self.cube_qpos_adr
        )

        self.data.qpos[cube_q:cube_q+3] = cube_position

        self.data.qpos[cube_q + 3: cube_q+7] = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ])

        self.initial_cube_z = (
            cube_position[2]
        )

        self.data.ctrl[:7] = (
            self.home_arm
        )

        self.data.ctrl[7] = 255

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        cube_position_now = (
            self.data.body("cube").xpos.copy()
        )

        self.best_lift_height = 0.0

        self.best_goal_xy_distance = np.linalg.norm(
            cube_position_now[:2] - self.goal_position[:2]
        )
        self.previous_goal_xy_distance = (
        self.best_goal_xy_distance
)

        self.reached_goal_region = False

        self.best_place_z_error= np.inf

        self.has_grasped = False
        self.has_lifted = False

        observation = (
            self._get_observation()
        )

        info = (self._get_task_info())

        return observation, info

    def step(self,action):

        action = np.asarray(action,dtype = np.float32)

        action = np.clip(action, -1.0, 1.0)

        current_q = (
            self.data.qpos[self.arm_qpos_ids].copy()
        )

        target_q = (
            current_q + self.action_scale * action[:7]
        )

        target_q = np.clip(target_q, self.lower_limits, self.upper_limits)

        self.data.ctrl[:7] = (
            target_q
        )

        gripper_ctrl =(
            action[7] + 1.0
        ) * 127.5

        self.data.ctrl[7] = np.clip(
            gripper_ctrl,
            0.0,
            255.0
        )

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)


        self.step_count += 1

        observation = (
            self._get_observation()
        )

        info = (self._get_task_info())

        reward = (
            self._compute_reward(info)
        )

        terminated = (
            info["is_success"]
        )

        truncated = (self.step_count >= self.max_episode_steps)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )