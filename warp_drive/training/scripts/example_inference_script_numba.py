# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Example training script for the grid world and continuous versions of Tag.
"""

import argparse
import logging
import os
import sys

import torch
import yaml

from example_envs.tag_continuous.tag_continuous import TagContinuous
from example_envs.tag_gridworld.tag_gridworld import CUDATagGridWorld, CUDATagGridWorldWithResetPool
from example_envs.single_agent.classic_control.cartpole.cartpole import CUDAClassicControlCartPoleEnv
from example_envs.single_agent.classic_control.mountain_car.mountain_car import CUDAClassicControlMountainCarEnv
from example_envs.single_agent.classic_control.continuous_mountain_car.continuous_mountain_car import \
    CUDAClassicControlContinuousMountainCarEnv
from example_envs.single_agent.classic_control.acrobot.acrobot import CUDAClassicControlAcrobotEnv
from example_envs.single_agent.classic_control.pendulum.pendulum import CUDAClassicControlPendulumEnv
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.trainers.trainer_a2c import TrainerA2C
from warp_drive.training.trainers.trainer_ddpg import TrainerDDPG
from warp_drive.training.utils.distributed_train.distributed_trainer_numba import (
    perform_distributed_training,
)
from warp_drive.training.utils.vertical_scaler import perform_auto_vertical_scaling
from warp_drive.utils.common import get_project_root

_ROOT_DIR = get_project_root()

_TAG_CONTINUOUS = "tag_continuous"
_TAG_GRIDWORLD = "tag_gridworld"
_TAG_GRIDWORLD_WITH_RESET_POOL = "tag_gridworld_with_reset_pool"

_CLASSIC_CONTROL_CARTPOLE = "single_cartpole"
_CLASSIC_CONTROL_MOUNTAIN_CAR = "single_mountain_car"
_CLASSIC_CONTROL_CONTINUOUS_MOUNTAIN_CAR = "single_continuous_mountain_car"
_CLASSIC_CONTROL_ACROBOT = "single_acrobot"
_CLASSIC_CONTROL_PENDULUM = "single_pendulum"


# Example usages (from the root folder):
# >> python warp_drive/training/example_training_script.py -e tag_gridworld
# >> python warp_drive/training/example_training_script.py --env tag_continuous

def recursive_update(base: dict, override: dict) -> dict:
    """Recursively update nested dicts."""
    for k, v in override.items():
        if (
            k in base
            and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def merge_yaml(base_path: str, override_path: str) -> dict:
    """
    Merge two YAML files: base and override.

    Args:
        base_path: path to base YAML file.
        override_path: path to override YAML file.

    Returns:
        merged (dict): merged configuration dictionary.
    """
    with open(base_path, "r", encoding="utf8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with open(override_path, "r", encoding="utf8") as f:
        override_cfg = yaml.safe_load(f) or {}

    merged = recursive_update(base_cfg, override_cfg)

    return merged


def setup_trainer_and_infer(
    run_configuration,
    list_of_states,
    use_argmax=True,
    device_id=0,
    num_devices=1,
    event_messenger=None,
    output_path=None,
    verbose=True,
):
    """
    Create the environment wrapper, define the policy mapping to agent ids,
    and create the trainer object. Also, perform training.
    """
    logging.getLogger().setLevel(logging.ERROR)

    num_envs = run_configuration["trainer"]["num_envs"]

    # Create a wrapped environment object via the EnvWrapper
    # Ensure that use_cuda is set to True (in order to run on the GPU)
    # ----------------------------------------------------------------
    if run_configuration["name"] == _TAG_CONTINUOUS:
        env_wrapper = EnvWrapper(
            TagContinuous(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _TAG_GRIDWORLD:
        env_wrapper = EnvWrapper(
            CUDATagGridWorld(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _TAG_GRIDWORLD_WITH_RESET_POOL:
        env_wrapper = EnvWrapper(
            CUDATagGridWorldWithResetPool(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _CLASSIC_CONTROL_CARTPOLE:
        env_wrapper = EnvWrapper(
            CUDAClassicControlCartPoleEnv(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _CLASSIC_CONTROL_MOUNTAIN_CAR:
        env_wrapper = EnvWrapper(
            CUDAClassicControlMountainCarEnv(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _CLASSIC_CONTROL_CONTINUOUS_MOUNTAIN_CAR:
        env_wrapper = EnvWrapper(
            CUDAClassicControlContinuousMountainCarEnv(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _CLASSIC_CONTROL_ACROBOT:
        env_wrapper = EnvWrapper(
            CUDAClassicControlAcrobotEnv(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _CLASSIC_CONTROL_PENDULUM:
        env_wrapper = EnvWrapper(
            CUDAClassicControlPendulumEnv(**run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            event_messenger=event_messenger,
            process_id=device_id,
        )
    else:
        raise NotImplementedError(
            f"Currently, the environments supported are ["
            f"{_TAG_GRIDWORLD}, "
            f"{_TAG_CONTINUOUS}"
            f"{_TAG_GRIDWORLD_WITH_RESET_POOL}"
            f"{_CLASSIC_CONTROL_CARTPOLE}"
            f"{_CLASSIC_CONTROL_MOUNTAIN_CAR}"
            f"{_CLASSIC_CONTROL_CONTINUOUS_MOUNTAIN_CAR}"
            f"{_CLASSIC_CONTROL_ACROBOT}"
            f"{_CLASSIC_CONTROL_PENDULUM}"
            f"]",
        )
    # Policy mapping to agent ids: agents can share models
    # The policy_tag_to_agent_id_map dictionary maps
    # policy model names to agent ids.
    # ----------------------------------------------------
    if len(run_configuration["policy"].keys()) == 1:
        # Using a single (or shared policy) across all agents
        policy_name = list(run_configuration["policy"])[0]
        if "tag_" in run_configuration["name"]:
            policy_tag_to_agent_id_map = {
                policy_name: list(env_wrapper.env.taggers) + list(env_wrapper.env.runners)
            }
        elif "single_" in run_configuration["name"]:
            policy_tag_to_agent_id_map = {
                policy_name: list(env_wrapper.env.agents)
            }
    else:
        # Using different policies for different (sets of) agents
        if "tag_" in run_configuration["name"]:
            policy_tag_to_agent_id_map = {
                "tagger": list(env_wrapper.env.taggers),
                "runner": list(env_wrapper.env.runners),
            }
        else:
            raise NotImplementedError
    # Assert that all the valid policies are mapped to at least one agent
    assert set(run_configuration["policy"].keys()) == set(
        policy_tag_to_agent_id_map.keys()
    )
    # Trainer object
    # --------------
    first_policy_name = list(run_configuration["policy"])[0]
    if run_configuration["policy"][first_policy_name]["algorithm"] == "DDPG":
        trainer = TrainerDDPG(
            env_wrapper=env_wrapper,
            config=run_configuration,
            policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
            device_id=device_id,
            num_devices=num_devices,
            verbose=verbose,
            inference_mode=True,
        )
    else:
        trainer = TrainerA2C(
            env_wrapper=env_wrapper,
            config=run_configuration,
            policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
            device_id=device_id,
            num_devices=num_devices,
            verbose=verbose,
            inference_mode=True,
        )

    # Perform training
    # ----------------
    episode_states_map, episode_actions_map, episode_rewards_map = \
        trainer.fetch_episode_states_multiple_envs(
        list_of_states=list_of_states,
        include_rewards_actions=True,
        use_argmax=use_argmax,
    )
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    np.savez_compressed(
        f"{output_path}/inference_data_states.npz",
        **episode_states_map,
    )
    np.savez_compressed(
        f"{output_path}/inference_data_actions.npz",
        **episode_actions_map,
    )
    np.savez_compressed(
        f"{output_path}/inference_data_rewards.npz",
        **episode_rewards_map,
    )

    trainer.graceful_close()
    perf_stats = trainer.perf_stats


if __name__ == "__main__":

    num_gpus_available = torch.cuda.device_count()
    assert num_gpus_available > 0, "The training script needs a GPU machine to run!"

    # Set logger level e.g., DEBUG, INFO, WARNING, ERROR\n",
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        help="the environment to infer. This also refers to the"
        "yaml file name in run_configs/.",
    )
    parser.add_argument(
        "--inference_config_path",
        type=str,
    )
    parser.add_argument(
        "--states",
        type=lambda s: s.split(","),
        help="comma-separated list of state names for inference"
    )
    parser.add_argument(
        "--use_argmax",
        type=bool,
        default=True,
        help="greedy way for inference"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="the inference output path"
    )

    args = parser.parse_args()
    assert args.env is not None, (
        "No env specified. Please use the '-e'- or '--env' option "
        "to specify an environment. The environment name should "
        "match the name of the yaml file in training/run_configs/."
    )

    # Read the run configurations specific to each environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(
        _ROOT_DIR, "warp_drive", "training", "run_configs", f"{args.env}.yaml"
    )
    if args.inference_config_path:
        inference_path = args.inference_config_path
    else:
        inference_path = os.path.join(
            _ROOT_DIR, "warp_drive", "training", "run_configs", f"{args.env}_inference.yaml"
        )
    assert args.output_path is not None, "The output path is not specified"

    if not os.path.exists(config_path):
        raise ValueError(
            "Invalid environment specified! The environment name should "
            "match the name of the yaml file in training/run_configs/."
        )
    if not os.path.exists(inference_path):
        raise ValueError(
            "Invalid inference specified! The environment name should "
            "match the name of the yaml file in training/run_configs/."
        )

    run_config = merge_yaml(base_path=config_path, override_path=inference_path)
    run_config["trainer"]["num_gpus"] = 1
    print(f"Inference with {run_config['trainer']['num_gpus']} GPU(s).")
    setup_trainer_and_infer(
        run_config,
        list_of_states=args.states,
        use_argmax=args.use_argmax,
        output_path=args.output_path,
    )
