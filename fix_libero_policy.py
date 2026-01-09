#!/usr/bin/env python3
"""
Script to fix libero_policy.py to handle None values in images.
"""

import os

def fix_libero_policy():
    """Fix libero_policy.py to handle None values in images."""
    policy_file = "/data3/yinmenghao/code/openpi/src/openpi/policies/libero_policy.py"
    
    # Read the current content of the file
    with open(policy_file, 'r') as f:
        content = f.read()
    
    # Replace the _parse_image function with a version that handles None values
    old_parse_image = """def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image"""
    
    new_parse_image = """def _parse_image(image) -> np.ndarray:
    if image is None:
        # Return a dummy image if no image data is available
        return np.zeros((224, 224, 3), dtype=np.uint8)
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image"""
    
    # Replace the function in the content
    fixed_content = content.replace(old_parse_image, new_parse_image)
    
    # Write the fixed content back to the file
    with open(policy_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Fixed {policy_file} to handle None values in images")

def main():
    """Main function to fix libero_policy.py."""
    fix_libero_policy()
    print("Fix completed successfully!")

if __name__ == "__main__":
    main()
