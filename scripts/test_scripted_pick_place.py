import time

import numpy as np
import mujoco
import mujoco.viewer

from envs.panda_pick_place_env import PandaPickPlaceEnv

env = PandaPickPlaceEnv(
    max_episode_steps=200,
    frame_skip=10,
    action_scale=0.04,
    randomize_cube=False
)

env.reset(seed=42)

site_id = (
    env.model.site("grasp_site").id
)

def simulation_steps(viewer, num_steps):

    for _ in range(num_steps):

        mujoco.mj_step(
            env.model,
            env.data
        )

        viewer.sync()

def move_grasp_site(
        viewer,
        target_position,
        tolerance = 0.007,
        max_iterations = 300
):
    for iteration in range(max_iterations):

        current_position = (
            env.data.site("grasp_site").xpos.copy()
        )

        error = (
            target_position - current_position
        )

        position_error = np.linalg.norm(
            error
        )

        if position_error < tolerance:

            print(
                f"Target reached | "
                f"error={position_error:.4f} m"
            )

            return True

        jac_position = np.zeros((3, env.model.nv))

        jac_rotation = np.zeros((3, env.model.nv))

        mujoco.mj_jacSite(
            env.model,
            env.data,
            jac_position,
            jac_rotation,
            site_id
        )

        J = jac_position[:, env.arm_dof_ids]

        damping = 0.05

        matrix = (
            J @ J.T + damping**2 * np.eye(3)
        )

        dq = (
            J.T @ np.linalg.solve(
                matrix,error
            )
        )

        dq = np.clip(
            dq,
            -0.015,
            0.015
        )

        current_q = (
            env.data.qpos[env.arm_qpos_ids].copy()
        )

        target_q = (
            current_q + dq
        )

        target_q = np.clip(
            target_q,
            env.lower_limits,
            env.upper_limits
        )

        env.data.ctrl[:7] = (
            target_q
        )

        simulation_steps(viewer,10)

    final_error = np.linalg.norm(
        target_position - env.data.site("grasp_site").xpos
    )

    print(
        f"Target not fully reached | "
        f"error={final_error:.4f} m"
    )

    return False


def print_stage(name):

    info = env._get_task_info()

    reward = env._compute_reward(info)

    print(
        f"\n{name}"
    )

    print(
        f"distance: "
        f"{info['distance']:.3f} m"
    )

    print(
        f"goal distance: "
        f"{info['goal_xy_distance']:.3f} m"
    )

    print(
        f"lift: "
        f"{info['lift_height']:.3f} m"
    )

    print(
        f"contacts: "
        f"{info['left_contact']}, "
        f"{info['right_contact']}"
    )

    print(
        f"reward: "
        f"{reward:.3f}"
    )

    print(
        f"success: "
        f"{info['is_success']}"
    )

    return info

with mujoco.viewer.launch_passive(env.model,env.data) as viewer:

    cube_position = (
        env.data.body("cube").xpos.copy()
    )

    goal_position = (
        env.goal_position.copy()
    )

    print("\n--- SCRIPTED PICK-AND-PLACE ---")

    pregrasp_target = (
        cube_position + np.array([
            0.0,
            0.0,
            0.18
        ])
    )

    print("\nMoving to pre-grasp...")

    move_grasp_site(
        viewer,
        pregrasp_target
    )

    grasp_target = (
        cube_position + np.array([
            0.0,
            0.0,
            0.01
        ])
    )

    print("\nDescending to cube...")

    move_grasp_site(
        viewer,
        grasp_target
    )

    print("\nClosing gripper...")

    env.data.ctrl[7] = 0.0

    simulation_steps(
        viewer,
        int(1.5 / env.model.opt.timestep)
    )

    print_stage(
        "AFTER GRASP"
    )

    np.savez(
    "models/pick_place_grasped_start.npz",

    qpos=env.data.qpos.copy(),
    qvel=env.data.qvel.copy(),
    ctrl=env.data.ctrl.copy()
)

    print(
        "\nGrasped start state saved!"
    )

    current_site = (
        env.data.site(
            "grasp_site"
        ).xpos.copy()
    )

    lift_target = (
        current_site + np.array([
            0.0,
            0.0,
            0.03
        ])
    )

    print("\nLifting cube...")

    move_grasp_site(
        viewer, lift_target
    )
    np.savez(
    "models/pick_place_lifted_3cm_start.npz",

    qpos=env.data.qpos.copy(),
    qvel=env.data.qvel.copy(),
    ctrl=env.data.ctrl.copy()
)

    print(
        "\n3 cm lifted start state saved!"
    )

    print_stage("AFTER LIFT")

    np.savez(
    "models/pick_place_lifted_start.npz",

    qpos=env.data.qpos.copy(),
    qvel=env.data.qvel.copy(),
    ctrl=env.data.ctrl.copy()
)

    print( 
    "\nLifted start state saved!"
)


    current_site = (
        env.data.site("grasp_site").xpos.copy()
    )

    transport_target = np.array([
        goal_position[0],
        goal_position[1],
        current_site[2]
    ])

    print("\nMoving toward target...")

    move_grasp_site(
        viewer,
        transport_target
    )

    print_stage("AFTER TRANSPORT")


    place_target = np.array([
        goal_position[0],
        goal_position[1],
        goal_position[2] + 0.01
    ])

    print("\nLowering cube...")

    move_grasp_site(viewer, place_target)

    print_stage("AFTER LOWER")

    print("\nOpening gripper...")

    env.data.ctrl[7] = 255

    simulation_steps(viewer,int(
        1.5 / env.model.opt.timestep
    ))

    final_info = print_stage(
        "FINAL"
    )

    print("\n-----------------------------")

    if final_info["is_success"]:

        print(
            "PICK-AND-PLACE SUCCESS!"
        )

    else:

        print(
            "Pick-and-place not successful yet."
        )

    print("-----------------------------")


    time.sleep(2)

env.close()









    



        