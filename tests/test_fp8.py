# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import unittest

import torch

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.test_utils import (
    get_launch_command,
    require_cuda_or_hpu,
    require_huggingface_suite,
    require_multi_device,
    require_torchao,
    require_transformer_engine,
    run_first,
)
from accelerate.test_utils.testing import require_deepspeed, run_command
from accelerate.utils import (
    AORecipeKwargs,
    FP8RecipeKwargs,
    has_ao_layers,
    has_transformer_engine_layers,
    is_torchao_available,
    is_transformer_engine_available,
)
from accelerate.utils.dataclasses import FP8BackendType


def test_fp8_backend_choices_consistency():
    """Test that fp8_backend choices are consistent across the codebase."""
    # Import the launch command parser to get the choices
    from accelerate.commands.launch import launch_command_parser
    from accelerate.commands.config.config_utils import _convert_fp8_backend
    
    parser = launch_command_parser()
    fp8_backend_action = None
    
    # Find the fp8_backend argument
    for action in parser._actions:
        if action.dest == "fp8_backend":
            fp8_backend_action = action
            break
    
    assert fp8_backend_action is not None, "fp8_backend argument not found in launch parser"
    
    # Get the choices from the launch command
    launch_choices = fp8_backend_action.choices
    
    # Get the enum values
    enum_values = [choice.value for choice in FP8BackendType]
    
    # Convert launch choices to uppercase to match enum values
    launch_choices_upper = [choice.upper() for choice in launch_choices]
    
    # Check that all launch choices are in the enum
    for choice in launch_choices_upper:
        assert choice in enum_values, f"Launch choice '{choice}' not found in FP8BackendType enum"
    
    # Check that all enum values are in launch choices
    for enum_value in enum_values:
        assert enum_value in launch_choices_upper, f"Enum value '{enum_value}' not found in launch choices"
    
    # Test the conversion function
    for i, choice in enumerate(launch_choices):
        converted = _convert_fp8_backend(i)
        assert converted == choice.upper(), f"Conversion failed for choice '{choice}'"


def can_convert_te_model():
    accelerator_kwargs = {"mixed_precision": "fp8", "kwargs_handlers": [FP8RecipeKwargs(backend="TE")]}
    accelerator = Accelerator(**accelerator_kwargs)
    dataloader = torch.utils.data.DataLoader(torch.randn(10, 32), batch_size=2)
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    assert has_transformer_engine_layers(model)


def maintain_proper_deepspeed_config(expected_version):
    assert AcceleratorState().deepspeed_plugin.zero_stage == expected_version, (
        f"Expected zero stage {expected_version} but got {AcceleratorState().deepspeed_plugin.zero_stage}"
    )


def can_convert_ao_model():
    from transformers import AutoModelForSequenceClassification

    accelerator_kwargs = {"mixed_precision": "fp8", "kwargs_handlers": [AORecipeKwargs()]}
    accelerator = Accelerator(**accelerator_kwargs)
    dataloader = torch.utils.data.DataLoader(torch.randn(10, 32), batch_size=2)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    assert has_ao_layers(model)


class TestFP8BackendChoices(unittest.TestCase):
    """Test that fp8_backend choices are consistent across the codebase."""
    
    def test_fp8_backend_choices_consistency(self):
        test_fp8_backend_choices_consistency()


@run_first
@require_transformer_engine
@require_cuda_or_hpu
class TestTransformerEngine(unittest.TestCase):
    def test_can_prepare_model_single_gpu(self):
        command = get_launch_command(num_processes=1, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8"]
        run_command(command)

    @require_multi_device
    def test_can_prepare_model_multi_gpu(self):
        command = get_launch_command(num_processes=2, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8"]
        run_command(command)

    @require_deepspeed
    @require_multi_device
    def test_can_prepare_model_multigpu_deepspeed(self):
        for zero_stage in [1, 2, 3]:
            os.environ["ZERO_STAGE"] = str(zero_stage)
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": zero_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": 1,
                "gradient_clipping": "auto",
                "steps_per_print": 2000,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False,
            }

            ds_config = json.dumps(ds_config)

            command = get_launch_command(
                num_processes=2, monitor_interval=0.1, use_deepspeed=True, deepspeed_config_file=ds_config
            )
            command += ["-m", "tests.test_fp8"]
            run_command(command)


@require_torchao
@require_huggingface_suite
class TestTorchAO(unittest.TestCase):
    def test_can_prepare_model_single_accelerator(self):
        command = get_launch_command(num_processes=1, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8"]
        run_command(command)

    @require_multi_device
    def test_can_prepare_model_multi_accelerator(self):
        command = get_launch_command(num_processes=2, monitor_interval=0.1)
        command += ["-m", "tests.test_fp8"]
        run_command(command)

    @require_deepspeed
    @require_multi_device
    def test_can_prepare_model_multi_accelerator_deepspeed(self):
        for zero_stage in [1, 2, 3]:
            os.environ["ZERO_STAGE"] = str(zero_stage)
            ds_config = {
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": zero_stage,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                },
                "gradient_accumulation_steps": 1,
                "gradient_clipping": "auto",
                "steps_per_print": 2000,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False,
            }

            ds_config = json.dumps(ds_config)

            command = get_launch_command(
                num_processes=2, monitor_interval=0.1, use_deepspeed=True, deepspeed_config_file=ds_config
            )
            command += ["-m", "tests.test_fp8"]
            run_command(command)


if __name__ == "__main__":
    # Test fp8 backend choices consistency
    test_fp8_backend_choices_consistency()
    
    # TE suite
    if is_transformer_engine_available():
        can_convert_te_model()
        if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true":
            maintain_proper_deepspeed_config(int(os.environ.get("ZERO_STAGE")))
    # AO suite
    if is_torchao_available():
        can_convert_ao_model()
