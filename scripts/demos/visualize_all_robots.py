import argparse
import sys
import torch
import os
import subprocess
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize all 12 robots in one scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# Quadruped configs
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG, ANYMAL_C_CFG, ANYMAL_B_CFG
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG, UNITREE_B2_CFG
from isaaclab_assets.robots.spot import SPOT_CFG

# Humanoid configs
from isaaclab_assets.robots.cassie import CASSIE_CFG
from isaaclab_assets.robots.agility import DIGIT_V4_CFG
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, H1_MINIMAL_CFG

@configclass
class UniversalSceneCfg(InteractiveSceneCfg):
    num_envs = 4
    env_spacing = 8.0

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # Light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # Global camera to capture the grid
    camera = CameraCfg(
        prim_path="/World/camera",
        update_period=0.02, # 50 fps
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # Position the camera at (18.0, 18.0, 12.0) looking at (2.0, 3.0, 0.0)
        offset=CameraCfg.OffsetCfg(
            pos=(18.0, 18.0, 12.0), 
            rot=(0.3420, 0.2027, 0.4679, 0.7893), # w, x, y, z
            convention="opengl"
        ),
    )

    # 12 Robots
    anymal_d = ANYMAL_D_CFG.replace(prim_path="/World/envs/env_.*/anymal_d")
    anymal_c = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/anymal_c")
    anymal_b = ANYMAL_B_CFG.replace(prim_path="/World/envs/env_.*/anymal_b")
    unitree_a1 = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/unitree_a1")
    unitree_go1 = UNITREE_GO1_CFG.replace(prim_path="/World/envs/env_.*/unitree_go1")
    unitree_go2 = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/unitree_go2")
    unitree_b2 = UNITREE_B2_CFG.replace(prim_path="/World/envs/env_.*/unitree_b2")
    spot = SPOT_CFG.replace(prim_path="/World/envs/env_.*/spot")
    
    cassie = CASSIE_CFG.replace(prim_path="/World/envs/env_.*/cassie")
    digit = DIGIT_V4_CFG.replace(prim_path="/World/envs/env_.*/digit")
    g1 = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/g1")
    h1 = H1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/h1")

def main():
    scene_cfg = UniversalSceneCfg()
    
    # The configs are set up to be placed in a grid in env_0 and then cloned.
    # Grid: 4 columns, 3 rows
    # x goes 0, 2, 4
    # y goes 0, 2, 4, 6
    configs = [
        scene_cfg.anymal_d, scene_cfg.anymal_c, scene_cfg.anymal_b, scene_cfg.unitree_a1,
        scene_cfg.unitree_go1, scene_cfg.unitree_go2, scene_cfg.unitree_b2, scene_cfg.spot,
        scene_cfg.cassie, scene_cfg.digit, scene_cfg.g1, scene_cfg.h1
    ]
    
    # We will orient them all to face -X so they look at the camera
    # Quaternion for 180 degree yaw (Z axis)
    import math
    yaw_180 = (0.0, 0.0, 0.0, 1.0) # w,x,y,z for 180 yaw is actually (0, 0, 0, 1) or (0, 0, 1, 0)
    # wait w=cos(90)=0, z=sin(90)=1 -> (0, 0, 0, 1)
    
    for i, cfg in enumerate(configs):
        row = i // 4
        col = i % 4
        cfg.init_state.pos = (row * 2.0, col * 2.0, 1.0)
        cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        
    # Reset simulation first
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    sim.set_camera_view([10, 10, 5], [3, 3, 0])
    
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(0.0)

    print("[INFO] Simulating and recording...")
    import cv2
    import numpy as np

    os.makedirs("tmp_frames", exist_ok=True)
    frames = []

    # Run for 200 steps (4 seconds at 50 FPS)
    for step in range(200):
        # Step physics (zero actions applied naturally)
        sim.step()
        scene.update(sim.cfg.dt)
        
        # Capture camera
        if step % 4 == 0:  # capture every 4 physics steps (dt=0.005 -> 0.02 render step)
            # image is (H, W, 4) from isaaclab camera
            img_tensor = scene.sensors["camera"].data.output["rgb"][0].cpu().numpy()
            
            # Convert RGBA to RGB if needed
            if img_tensor.shape[2] == 4:
                img_tensor = img_tensor[:, :, :3]
                
            # IsaacLab images are usually uint8 already
            if img_tensor.dtype != np.uint8:
                img_tensor = (img_tensor * 255).astype(np.uint8)
                
            frames.append(img_tensor)
            
    print(f"[INFO] Captured {len(frames)} frames. Generating GIF with ffmpeg...")
    
    # Write frames to tmp dir for ffmpeg
    for i, f in enumerate(frames):
        # Convert RGB to BGR for cv2 imwrite
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"tmp_frames/frame_{i:04d}.png", f_bgr)
        
    # Generate GIF
    out_path = "docs/source/_static/all_robots_unified.gif"
    cmd = [
        "ffmpeg",
        "-framerate", "50",
        "-i", "tmp_frames/frame_%04d.png",
        "-filter_complex", "split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer",
        "-y",
        out_path
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup
    for i in range(len(frames)):
        os.remove(f"tmp_frames/frame_{i:04d}.png")
    os.rmdir("tmp_frames")
    
    print(f"[SUCCESS] Unified GIF saved to {out_path}")

if __name__ == "__main__":
    main()
