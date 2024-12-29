# https://github.com/rohanpsingh/mujoco-python-viewer/issues/28
import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('/home/user/yzchen_ws/TAMP-ubuntu22/ALOHA/act/assets/bimanual_viperx_ee_insertion.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()