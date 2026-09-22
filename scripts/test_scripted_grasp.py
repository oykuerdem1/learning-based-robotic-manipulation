import time
import numpy as np

import mujoco
import mujoco.viewer

from envs.panda_grasp_scene import create_grasp_scene

model, data = create_grasp_scene()

arm_qpos_ids = np.array([
    model.joint(f"joint{i}").qposadr[0]
    for i in range(1,8)
])

arm_dof_ids = np.array([
    model.joint(f"joint{i}").dofadr[0]
    for i in range(1,8)
])

lower_limits = model.actuator_ctrlrange[:7,0]
upper_limits = model.actuator_ctrlrange[:7,1]

grasp_site_id = model.site("grasp_site").id


cube_body_id = model.body("cube").id

left_finger_body_id = model.body("left_finger").id
right_finger_body_id = model.body("right_finger").id


def physics_step(viewer):

    start = time.time()

    mujoco.mj_step(
        model, data
    )

    viewer.sync()

    remaining = (
        model.opt.timestep - (time.time() - start)
    )

    if remaining > 0:
        time.sleep(remaining)


def hold(viewer, seconds):

    num_steps = int(seconds / model.opt.timestep)

    for _ in range(num_steps):

        if not viewer.is_running():
            break

        physics_step(viewer)


def bodies_in_contact(
        body_a, body_b
):
    for i in range(data.ncon):

        contact = data.contact[i]

        body1 = model.geom_bodyid[contact.geom1]

        body2 = model.geom_bodyid[contact.geom2]

        if (
            body1 == body_a and body2 == body_b
        ) or ( body2 == body_a and body1 == body_b):
            return True
    return False

def move_grasp_site_to(
        viewer,
        target_position,
        gripper_ctrl=255,
        max_steps = 1000,
        tolerance = 0.007
):
    for step in range(max_steps):

        current_position = (
            data.site("grasp_site").xpos.copy()
        )

        position_error = (
            target_position - current_position
        )

        position_norm = np.linalg.norm(position_error)


        current_rotation = (
            data.site("grasp_site").xmat.reshape(3,3)
        )

        current_z = current_rotation[:,2]

        desired_z = np.array([
            0.0,
            0.0,
            -1.0
        ])

        cos_angle = np.clip(
            np.dot(
                current_z,
                desired_z
            ),
            -1.0,
            1.0
        )

        tilt = np.arccos(
            cos_angle
        )

        if step % 25 == 0:

           print(
                f"step={step:4d} | "
                f"pos_error={position_norm:.4f} | "
                f"tilt={np.degrees(tilt):.1f} deg"
            )
        

        if (position_norm < tolerance and tilt < np.deg2rad(10.0)):
            print("Target reached.")

            return True

        jac_position = np.zeros((3, model.nv))

        jac_rotation = np.zeros((3,model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jac_position,
            jac_rotation,
            grasp_site_id
        )

        J = jac_position[:, arm_dof_ids]

        damping = 0.05

        dq = (
            J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3),
                                  position_error)
        )

        dq = np.clip(
            dq,
            -0.02,
            0.02
        )

        current_q = (
            data.qpos[arm_qpos_ids].copy()
        )

        target_q = (
            current_q + dq
        )

        target_q = np.clip(target_q, lower_limits, upper_limits)

        data.ctrl[:7] = target_q

        data.ctrl[7] = gripper_ctrl

        for _ in range(10):
            physics_step(viewer)


    print("Target NOT reached.")

    return False

with mujoco.viewer.launch_passive(model, data) as viewer:

    for _ in range(250):
        physics_step(viewer)


    cube_position = (
        data.body("cube").xpos.copy()
    )

    initial_cube_z = cube_position[2]

    print(
        "\nCube position:",
        cube_position
    )

    pregrasp_position = (
        cube_position + np.array([0.0,0.0,0.18])
    )


    print(
        "\n--- PRE-GRASP ---"
    )

    pregrasp_success = (
        move_grasp_site_to(
            viewer, pregrasp_position,gripper_ctrl=255
        )
    )

    print(
        "Pre-grasp success:",
        pregrasp_success
    )


    grasp_position = (
        cube_position + np.array([
            0.0,
            0.0,
            0.01
        ])
    )

    print(
        "\n--- DESCEND ---"
    )

    descend_success = (
        move_grasp_site_to(viewer, grasp_position, gripper_ctrl=255, max_steps=1000)

    )

    print(
        "Descend success:",
        descend_success
    )

    print(
        "\n--- CLOSE GRIPPER ---"
    )

    data.ctrl[7] = 0

    hold(viewer, seconds = 2.0)

    left_contact = (
        bodies_in_contact(cube_body_id,left_finger_body_id)
    )

    right_contact = (
        bodies_in_contact(
            cube_body_id, right_finger_body_id
        )
    )

    print(
        "Left finger contact:",
        left_contact
    )

    print(
        "Right finger contact:",
        right_contact
    )

    print(
        "Cube before lift:",
        data.body("cube").xpos.copy()
    )

    current_site = (
        data.site("grasp_site").xpos.copy()
    )

    lift_position = (
        current_site + np.array([0.0, 0.0, 0.12])
    )

    print(
        "\n--- LIFT ---"
    )

    lift_success = (
        move_grasp_site_to(
            viewer, lift_position,

            gripper_ctrl=0,
            max_steps=1000
        )
    )

    cube_after = (
        data.body("cube").xpos.copy()
    )

    print(
        "\nLift target reached:",
        lift_success
    )

    print(
        "Cube after lift:",
        cube_after
    )

    print(
        "Cube height increase:",
        cube_after[2]
        - initial_cube_z
    )

    if (cube_after[2] > initial_cube_z + 0.05):
        print(
            "\n======================="
        )

        print(
            "GRASP SUCCESS!"
        )

        print(
            "Cube was lifted."
        )

        print(
            "======================="
        )
    else:

        print(
            "\n======================="
        )

        print(
            "GRASP FAILED"
        )

        print(
            "Cube was not lifted."
        )

        print(
            "======================="
        )
    hold(viewer, seconds = 3.0)
    

