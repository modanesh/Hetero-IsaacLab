# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import math
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--record_trajectory", action="store_true", default=False, help="Record the trajectory of the agent.")
parser.add_argument("--cluster_robots", action="store_true", default=False, help="Cluster all robots into a cinematic grid on a specific terrain.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    gym_make_kwargs = {}
    first_robot = "anymal_d"  # Default fallback
    if args_cli.quadrupeds:
        quadrupeds_list = [name.strip() for name in args_cli.quadrupeds.split(',')]
        gym_make_kwargs["quadrupeds"] = quadrupeds_list
        if len(quadrupeds_list) > 0:
            first_robot = quadrupeds_list[0]
    elif args_cli.humanoids:
        humanoids_list = [name.strip() for name in args_cli.humanoids.split(',')]
        gym_make_kwargs["humanoids"] = humanoids_list
        if len(humanoids_list) > 0:
            first_robot = humanoids_list[0]


    # Track the first robot dynamically only if clustering is enabled
    if args_cli.cluster_robots and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = first_robot
        env_cfg.viewer.eye = (10.0, 10.0, 10.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.resolution = (1920, 1080)  # 1080p resolution (can also use 3840, 2160 for 4K)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None, **gym_make_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Cleanly cluster all robots around a specific terrain block
    if args_cli.cluster_robots and args_cli.num_envs and args_cli.num_envs > 1:
        if hasattr(env.unwrapped, "scene") and hasattr(env.unwrapped.scene, "env_origins"):
            origins = env.unwrapped.scene.env_origins.clone()
            
            # --- TERRAIN SELECTION ---
            # Columns 0-3: pyramid_stairs
            # Columns 4-7: pyramid_stairs_inv
            # Columns 8-11: boxes (vertically shifted cubes)
            # Columns 12-15: random_rough
            # Columns 16-17: hf_pyramid_slope
            # Columns 18-19: hf_pyramid_slope_inv
            
            if hasattr(env.unwrapped.scene, "terrain") and hasattr(env.unwrapped.scene.terrain, "terrain_origins"):
                terrain_origins = env.unwrapped.scene.terrain.terrain_origins
                
                if terrain_origins is not None:
                    # Safely get dimensions to prevent Out Of Bounds errors if the terrain generator changes
                    max_rows = terrain_origins.shape[0]
                    max_cols = terrain_origins.shape[1]
                    
                    target_difficulty_row = min(3, max_rows - 1)  # 0 to max (0 is flat, max is hardest)
                    target_terrain_col = min(9, max_cols - 1)     # 8-11 are the boxes!
                    
                    base_origin = terrain_origins[target_difficulty_row, target_terrain_col].clone()
                else:
                    base_origin = origins[0].clone()
            else:
                base_origin = origins[0].clone()
                
            # Arrange environments in a dynamic square grid around the chosen terrain block
            
            grid_size = math.ceil(math.sqrt(args_cli.num_envs))
            
            for i in range(args_cli.num_envs):
                row = i // grid_size
                col = i % grid_size
                origins[i, 0] = base_origin[0] + row * 1.0
                origins[i, 1] = base_origin[1] + col * 1.0
                origins[i, 2] = base_origin[2]
            
            # Disable curriculum so IsaacLab doesn't overwrite our custom origins during reset
            if hasattr(env.unwrapped.scene, "terrain") and hasattr(env.unwrapped.scene.terrain, "cfg"):
                if getattr(env.unwrapped.scene.terrain.cfg, "terrain_generator", None) is not None:
                    env.unwrapped.scene.terrain.cfg.terrain_generator.curriculum = False
            
            env.unwrapped.scene.env_origins[:] = origins
            env.reset() # Apply the new origins to the simulation

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename=f"policy_{resume_path.split('/')[-2]}.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename=f"policy_{resume_path.split('/')[-2]}.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"policy_{resume_path.split('/')[-2]}.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"policy_{resume_path.split('/')[-2]}.onnx")

    dt = env.unwrapped.step_dt

    if args_cli.record_trajectory:
        obs_list, act_list, rew_list, done_list = [], [], [], []

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            next_obs, rwds, dns, infos = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dns)
            else:
                policy_nn.reset(dns)

        if args_cli.record_trajectory:
            if len(obs_list) >= args_cli.video_length:
                print(f"[INFO] Reached maximum trajectory length of {args_cli.video_length} steps. Stopping recording.")
                break
            obs_list.append(obs)
            act_list.append(actions)
            rew_list.append(rwds)
            done_list.append(dns)

        obs = next_obs

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if args_cli.record_trajectory and len(obs_list) >= 1000:
            print(f"[INFO] Collected {len(obs_list)} timesteps, stopping the simulation.")
            break

    # close the simulator
    env.close()

    if args_cli.record_trajectory:
        all_observations = torch.stack(obs_list, dim=0)  # [T, num_envs, obs_dim]
        all_actions = torch.stack(act_list, dim=0)
        all_rewards = torch.stack(rew_list, dim=0)
        all_dones = torch.stack(done_list, dim=0)
        # save all as a torch tensor in a single pt file
        print(f"[INFO] Saving trajectories to {export_model_dir}/trajectories_{resume_path.split('/')[-2]}.pt")
        print(f"[INFO] Observations shape: {all_observations.shape}")
        torch.save(
            {
                "observations": all_observations,
                "actions": all_actions,
                "rewards": all_rewards,
                "dones": all_dones,
            },
            os.path.join(export_model_dir, f"trajectories_{resume_path.split('/')[-2]}.pt"),
        )



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
