import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    if image is None:
        # Return a dummy image if no image data is available
        return np.zeros((224, 224, 3), dtype=np.uint8)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AssemblyBunOutputs(transforms.DataTransformFn):
    """
    Output transform for Assembly Bun dataset (8D actions).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 8 actions (7 DOF + 1 Gripper)
        return {"actions": np.asarray(data["actions"][:, :8])}


@dataclasses.dataclass(frozen=True)
class StackBowlsInputs(transforms.DataTransformFn):
    """
    Input transform for Stack Bowls dataset (7D actions, axis-angle rotation).

    Expected input keys (set via RepackTransform during training, or sent directly during inference):
        - observation/front_image: front camera RGB image
        - observation/wrist_image: wrist camera RGB image
        - observation/state: robot state [x, y, z, ax, ay, az, gripper]
        - actions: (training only) action targets
        - prompt: language instruction
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images: LeRobot stores video frames as float32 (C,H,W), convert to uint8 (H,W,C).
        front_image = _parse_image(data["observation/front_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                # Map to model's internal image slots:
                # base_0_rgb = third-person / front camera
                # left_wrist_0_rgb = wrist camera
                # right_wrist_0_rgb = not used, padded with zeros
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class StackBowlsOutputs(transforms.DataTransformFn):
    """
    Output transform for Stack Bowls dataset (7D actions).
    Action format: [x, y, z, ax, ay, az, gripper] (axis-angle rotation).
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 7 actions (3 pos + 3 axis-angle + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class AssemblyThingsInputs(transforms.DataTransformFn):
    """
    Input transform for Assembly Things multi-task dataset (7D actions, axis-angle rotation).

    Same camera/action format as StackBowlsInputs:
        - observation/front_image: front camera RGB image
        - observation/wrist_image: wrist camera RGB image
        - observation/state: robot state [x, y, z, ax, ay, az, gripper]
        - actions: (training only) action targets
        - prompt: language instruction (varies per task)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/front_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AssemblyThingsOutputs(transforms.DataTransformFn):
    """
    Output transform for Assembly Things multi-task dataset (7D actions).
    Action format: [x, y, z, ax, ay, az, gripper] (axis-angle rotation).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class PlacePhoneInputs(transforms.DataTransformFn):
    """
    Input transform for Place Phone dataset (7D actions, axis-angle rotation).

    Same camera/action format as StackBowlsInputs:
        - observation/front_image: front camera RGB image
        - observation/wrist_image: wrist camera RGB image
        - observation/state: robot state [x, y, z, ax, ay, az, gripper]
        - actions: (training only) action targets
        - prompt: language instruction
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/front_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PlacePhoneOutputs(transforms.DataTransformFn):
    """
    Output transform for Place Phone dataset (7D actions).
    Action format: [x, y, z, ax, ay, az, gripper] (axis-angle rotation).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
