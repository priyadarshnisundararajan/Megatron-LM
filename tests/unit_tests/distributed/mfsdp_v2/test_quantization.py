# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Quantized training tests for the experimental Megatron-FSDP path."""

import pytest
import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from transformer_engine.pytorch.optimizers import FusedAdam

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import BlockAtomic
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
    effective_dtype,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy


def _check_mxfp8_training_against_reference(
    model, reference, recipe, parameter_placement, distributed_setup
):
    """Compare three distributed Adam steps with an independently updated TE model."""
    device = distributed_setup.device
    reference_parameters = dict(reference.named_parameters())
    reference_main_weights = {}
    for name, parameter in model.named_parameters():
        initial_value = (
            parameter.get_high_precision_init_val()
            if effective_dtype(parameter) == torch.uint8
            else parameter.detach()
        )
        reference_main_weights[name] = nn.Parameter(
            initial_value.to(device=device, dtype=torch.float32).clone()
        )

    def sync_reference_weights():
        with torch.no_grad():
            for name, parameter in reference_parameters.items():
                if effective_dtype(parameter) == torch.uint8:
                    parameter.quantize_(reference_main_weights[name])
                else:
                    parameter.copy_(reference_main_weights[name])

    sync_reference_weights()
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    placements = Placements(
        dp_axes=[0], parameter=[parameter_placement], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device):
        fully_shard(
            model,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )

    sharded_parameters = dict(model.named_parameters())
    reference_slices = {}
    for name, parameter in sharded_parameters.items():
        # Packed buffers can give a parameter uneven or empty shards. Derive the
        # reference slices from the actual shard sizes, independently of DBuffer's layout.
        shard_rows = [None] * distributed_setup.world_size
        dist.all_gather_object(shard_rows, parameter.to_local().shape[0])
        assert sum(shard_rows) == reference_main_weights[name].shape[0]
        start = sum(shard_rows[: distributed_setup.rank])
        reference_slices[name] = slice(start, start + shard_rows[distributed_setup.rank])
        torch.testing.assert_close(
            parameter.to_local(),
            reference_main_weights[name][reference_slices[name]],
            rtol=0,
            atol=0,
        )

    optimizer = FusedAdam(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = FusedAdam(list(reference_main_weights.values()), lr=0.01)
    # Different data on each rank makes an omitted or incorrect gradient reduction observable.
    torch.manual_seed(1234 + distributed_setup.rank)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = torch.randn(32, 64, dtype=torch.bfloat16, device=device)
        with te.autocast(recipe=recipe):
            output = model(x)
            reference_output = reference(x)
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        target = torch.randn_like(output)
        loss = (output.float() - target.float()).square().mean()
        reference_loss = (reference_output.float() - target.float()).square().mean()
        torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
        loss.backward()
        reference_loss.backward()

        for name, reference_parameter in reference_parameters.items():
            # The unsharded reference averages full gradients independently of
            # MFSDP's packed reduce-scatter, retaining the BF16 reduction dtype.
            dist.all_reduce(reference_parameter.grad, op=dist.ReduceOp.AVG)
            torch.testing.assert_close(
                sharded_parameters[name].grad.to_local(),
                reference_parameter.grad[reference_slices[name]],
                rtol=0,
                atol=0,
                msg=lambda msg, name=name, step=step: f"{name} gradient at step {step}: {msg}",
            )
            reference_main_weights[name].grad = reference_parameter.grad.float()

        optimizer.step()
        reference_optimizer.step()
        for name, parameter in sharded_parameters.items():
            torch.testing.assert_close(
                parameter.to_local(),
                reference_main_weights[name][reference_slices[name]],
                rtol=0,
                atol=0,
                msg=lambda msg, name=name, step=step: f"{name} main weight at step {step}: {msg}",
            )
        sync_reference_weights()
        # Check the freshly requantized weights, including after the final update.
        with torch.no_grad(), te.autocast(recipe=recipe):
            torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)


@pytest.mark.parametrize("parameter_placement", [Shard(0), Replicate()])
def test_mxfp8_linear_training_step_uses_quantized_dbuffer(distributed_setup, parameter_placement):
    """ZeRO-1/3 MXFP8 training matches an unsharded numerical reference over three steps."""
    te = pytest.importorskip("transformer_engine")
    if distributed_setup.world_size != 2:
        pytest.skip("MXFP8 grouped DBuffer coverage requires exactly two ranks.")
    if torch.cuda.get_device_capability(distributed_setup.device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")

    torch.manual_seed(2026)
    recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=te.common.recipe.Format.HYBRID)
    with te.pytorch.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        model = te.pytorch.Linear(
            64, 256, bias=False, params_dtype=torch.bfloat16, device=distributed_setup.device
        )
        reference = te.pytorch.Linear(
            64, 256, bias=False, params_dtype=torch.bfloat16, device=distributed_setup.device
        )
    _check_mxfp8_training_against_reference(
        model, reference, recipe, parameter_placement, distributed_setup
    )
    parameter_group = model.parameter_groups[0]
    assert isinstance(parameter_group.model_weight, QuantizedDBuffer)
    assert isinstance(parameter_group.post_optimizer_model_weight, QuantizedDBuffer)
    assert (
        parameter_group.post_optimizer_model_weight is parameter_group.model_weight
    ) is isinstance(parameter_placement, Shard)
    assert parameter_group.main_weight.placements == (BlockAtomic(32),)


@pytest.mark.parametrize("parameter_placement", [Shard(0), Replicate()])
def test_mxfp8_quantized_dbuffer_handles_multiple_weights_and_biases(
    distributed_setup, parameter_placement
):
    """Grouped MXFP8 weights and BF16 biases match an unsharded MLP over three steps."""
    te = pytest.importorskip("transformer_engine")
    if distributed_setup.world_size != 2:
        pytest.skip("MXFP8 grouped DBuffer coverage requires exactly two ranks.")
    if torch.cuda.get_device_capability(distributed_setup.device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")

    def make_model():
        return nn.Sequential(
            te.pytorch.Linear(
                64, 128, bias=True, params_dtype=torch.bfloat16, device=distributed_setup.device
            ),
            nn.GELU(),
            te.pytorch.Linear(
                128, 32, bias=True, params_dtype=torch.bfloat16, device=distributed_setup.device
            ),
        )

    torch.manual_seed(2026)
    recipe = te.common.recipe.MXFP8BlockScaling(fp8_format=te.common.recipe.Format.HYBRID)
    with te.pytorch.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
        model = make_model()
        reference = make_model()
    _check_mxfp8_training_against_reference(
        model, reference, recipe, parameter_placement, distributed_setup
    )
    quantized_groups = [
        group
        for group in model.parameter_groups
        if isinstance(group.model_weight, QuantizedDBuffer)
    ]
    assert len(quantized_groups) == 1
    parameter_group = quantized_groups[0]
    assert parameter_group.model_weight.rowwise_data.layout.tensor_shapes == (
        torch.Size((128, 64)),
        torch.Size((32, 128)),
    )
    assert len(parameter_group.fsdp_parameters) == 2
    assert (
        parameter_group.post_optimizer_model_weight is parameter_group.model_weight
    ) is isinstance(parameter_placement, Shard)
    assert len(model.parameter_groups) == 2
