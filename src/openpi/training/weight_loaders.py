import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _try_truncate_weight(loaded: np.ndarray, expected_shape: tuple) -> np.ndarray | None:
    """Try to truncate a loaded weight to match the expected shape.

    For each dimension, if expected <= loaded, slice the first `expected` elements.
    This preserves pre-trained features for overlapping dimensions (e.g., when action_dim
    changes from 32 to 7, the first 7 dims' projections are reused).

    Returns the truncated array, or None if truncation is not possible (i.e., any expected
    dimension is larger than the loaded dimension).
    """
    if len(loaded.shape) != len(expected_shape):
        return None
    # Check that every expected dim is <= the loaded dim (can only shrink, not grow).
    for ld, ed in zip(loaded.shape, expected_shape):
        if ed > ld:
            return None
    # Build slicing tuple: take first `expected` elements along each axis.
    slices = tuple(slice(0, ed) for ed in expected_shape)
    return loaded[slices]


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    # When shapes mismatch (e.g., action_dim changed from 32 to 7), try to truncate the
    # loaded weight to reuse pre-trained features for the overlapping dimensions.
    # This is better than random init since position/rotation projections are preserved.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            if hasattr(v, "shape") and hasattr(ref, "shape") and v.shape != ref.shape:
                truncated = _try_truncate_weight(v, ref.shape)
                if truncated is not None:
                    logger.warning(
                        f"Truncating weight '{k}': {v.shape} -> {ref.shape} "
                        "(reusing pre-trained features for overlapping dimensions)."
                    )
                    result[k] = truncated.astype(ref.dtype) if truncated.dtype != ref.dtype else truncated
                else:
                    logger.warning(
                        f"Skipping weight '{k}': shape mismatch (loaded={v.shape}, expected={ref.shape}), "
                        "cannot truncate. This layer will be randomly initialized."
                    )
                    result[k] = ref  # keep reference (ShapeDtypeStruct) for tree structure
                continue
            result[k] = v.astype(ref.dtype) if v.dtype != ref.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
