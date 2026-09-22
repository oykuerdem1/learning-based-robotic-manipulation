import numpy as np

import mujoco
import mujoco_menagerie as menagerie

def create_grasp_scene(
        add_place_target=False
):

    spec = menagerie.get(
        "franka_emika_panda"
    ).spec()

    # --------------------------------------------------
    # Add a grasp site between the fingertips
    # --------------------------------------------------

    hand = spec.body("hand")

    hand.add_site(
        name="grasp_site",
        pos=[0.0, 0.0, 0.103],
        size=[0.008, 0.0, 0.0],
        rgba=[0.0, 1.0, 0.0, 1.0]
    )


    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos = [0.50, 0.0, 0.375],
        size = [0.35, 0.45, 0.025],
        rgba = [0.55, 0.55, 0.55,1.0],
        friction = [1.0, 0.005, 0.0001]
    )

    if add_place_target:

        spec.worldbody.add_geom(
            name="place_target",

            type = mujoco.mjtGeom.mjGEOM_CYLINDER,

            pos = [
                0.50,
                0.20,
                0.402
            ],

            size = [
                0.06,
                0.002
            ],

            rgba = [
                0.1,
                0.8,
                0.2,
                0.35
            ],

            contype = 0,
            conaffinity = 0
        )

    cube = spec.worldbody.add_body(
        name="cube",
        pos=[0.50, 0.0, 0.425]
    )

    cube.add_freejoint()

    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.025, 0.025,0.025],
        mass = 0.10,
        rgba = [0.9, 0.25,0.10,1.0],
        friction = [1.2,0.005,0.0001]
    )

    cube.add_site(
        name="cube_site",
        pos = [0.0, 0.0, 0.0],
        size = [0.008, 0.0, 0.0],
        rgba=[1.0, 1.0,0.0, 1.0]
    )

    model = spec.compile()
    data = mujoco.MjData(model)

    home_arm = np.array([
        0.0,
        0.0,
        0.0,
        -1.57079,
        0.0,
        1.57079,
        -0.7853
    ])

    for i in range(7):
        joint_name = f"joint{i + 1}"

        qpos_adress = (
            model.joint(joint_name).qposadr[0]
        )

        data.qpos[qpos_adress] = (
            home_arm[i]
        )


    for finger_name in ["finger_joint1", "finger_joint2"]:
        qpos_adress = (
            model.joint(finger_name).qposadr[0]
        )

        data.qpos[qpos_adress] = 0.04

    data.ctrl[:7] = home_arm
    data.ctrl[7] = 255

    mujoco.mj_forward(model,data)

    return model, data