import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte

class NLMTransform:
    """
    Custom PyTorch transform for applying Non-local Means denoising.
    Input is a PIL Image (grayscale).
    Output will be a PIL Image (grayscale) or a Tensor if placed after ToTensor.
    """
    def __init__(self, h=0.1, fast_mode=True, patch_size=7, patch_distance=15, multichannel=False):
        """
        Args:
            h (float): Parameter controlling the degree of filtering. A larger h will lead to
                       more aggressive denoising. It should be proportional to the noise standard deviation.
            fast_mode (bool): If True, use the fast NLM algorithm.
            patch_size (int): Size of patches (e.g., 7 for 7x7 patches).
            patch_distance (int): Maximum distance between patches to be compared.
            multichannel (bool): Set to True if the image has multiple channels (e.g., RGB).
                                 For grayscale mammograms, keep as False.
        """
        self.h = h
        self.fast_mode = fast_mode
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.multichannel = multichannel 

    def __call__(self, img):
        """
        Args:
            img (PIL Image or torch.Tensor): The image to denoise.
                                             If PIL Image, convert to numpy.
                                             If torch.Tensor, convert to numpy, ensure it's on CPU.
        Returns:
            PIL Image or torch.Tensor: The denoised image in the same format as input.
        """
        # Determine if the input is a PIL Image or a PyTorch Tensor
        is_pil_image = isinstance(img, Image.Image)
        if is_pil_image:
            # Convert PIL Image to numpy float,  NLM expects float in [0, 1]
            img_np = img_as_float(np.array(img))
        elif isinstance(img, torch.Tensor):
            # If input is a Tensor, already float and in [0,1]
            # Convert to numpy.
            # Handle potential 3-channel (repeated) tensor by taking one channel if multichannel=False
            if img.dim() == 3 and img.shape[0] == 3 and not self.multichannel:
                img_np = img[0, :, :].cpu().numpy() # Take the first channel
            else:
                img_np = img.squeeze().cpu().numpy() # Remove channel dim if 1, then to numpy
        else:
            raise TypeError(f"Input must be PIL Image or torch.Tensor. Got {type(img)}")

        # https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_nonlocal_means.html
        # We try with h=0.1 but can also try estimating.
        # sigma_est = estimate_sigma(img_np, average_sigmas=True, channel_axis=-1 if self.multichannel else None)
        # print(f"Estimated noise sigma: {sigma_est}")
        # Estimate h based on sigma:
        # self.h = self.h * sigma_est 

        # Apply NLM denoising
        denoised_img_np = denoise_nl_means(
            img_np,
            h=self.h,
            fast_mode=self.fast_mode,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            channel_axis=None # Should be None for grayscale
        )

        # Convert back to original format
        if is_pil_image:
            # Convert numpy float back to PIL Image 
            return Image.fromarray(img_as_ubyte(denoised_img_np), mode='L')
        else:
            # Convert numpy float back to PyTorch Tensor
            # Add channel dimension back if it was removed for NLM
            if not self.multichannel and denoised_img_np.ndim == 2:
                # If original was 3 channels repeated, repeat again.
                if img.dim() == 3 and img.shape[0] == 3:
                     return torch.from_numpy(denoised_img_np).unsqueeze(0).repeat(3, 1, 1).float()
                else: # Original was 1 channel
                    return torch.from_numpy(denoised_img_np).unsqueeze(0).float()
            else:
                return torch.from_numpy(denoised_img_np).float()

