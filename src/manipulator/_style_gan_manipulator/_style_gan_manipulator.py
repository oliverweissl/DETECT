from typing import Union

import numpy as np
import torch
from torch import Tensor

from src.manipulator._style_gan_manipulator._mix_candidate import CandidateList

from .._manipulator import Manipulator
from . import dnnlib
from ._load_stylegan import load_stylegan
from .torch_utils.ops import upfirdn2d


class StyleGANManipulator(Manipulator):
    """
    A class geared towards style-mixing using style layers in a StlyeGAN.

    This class is heavily influenced by the Renderer class in the StyleGAN3 repo.
    """

    _generator: torch.nn.Module
    _device: torch.device
    _has_input_transform: bool
    _noise_mode: str
    _interpolate: bool

    _mix_dims: Tensor  # Range for dimensions to mix styles with.

    def __init__(
        self,
        generator: Union[torch.nn.Module, str],
        device: torch.device,
        mix_dims: tuple[int, int],
        interpolate: bool = True,
        noise_mode: str = "random",
    ):
        """
        Initialize the StyleMixer object.

        :param generator: The generator network to use or the path to its pickle file.
        :param device: The torch device to use (should be cuda).
        :param mix_dims: The w-dimensions to use for mixing (range index).
        :param noise_mode: The noise mode to be used for generation (const, random).
        :param interpolate: Whether to interpolate style layers or mix.
        :raises ValueError: If `manipulation_mode` or `noise_mode` is not supported.
        """
        self._interpolate = interpolate

        if noise_mode in ["random", "const"]:
            self._noise_mode = noise_mode
        else:
            raise ValueError(f"Unknown noise mode: {noise_mode}")

        self._generator = (
            generator if isinstance(generator, torch.nn.Module) else load_stylegan(generator)
        )
        self._device = device
        self._generator.to(self._device)
        self._has_input_transform = hasattr(self._generator.synthesis, "input") and hasattr(
            self._generator.synthesis.input, "transform"
        )

        self._mix_dims = torch.arange(
            *mix_dims, device=self._device
        )  # Dimensions of w -> we only want to mix style layers.

    def manipulate(
        self,
        candidates: CandidateList,
        cond: list[int],
        weights: list[float],
        random_seed: int = 0,
    ) -> Tensor:
        """
        Generate data using style mixing or interpolation.

        This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

        :param candidates: The candidates used for style-mixing.
        :param cond: The manipulation conditions (layer combinations).
        :param weights: The weights for manipulating layers.
        :param random_seed: The seed for randomization.
        :returns: The generated image (C x H x W) as float with range [0, 255].
        """
        assert len(cond) == len(
            weights
        ), "Error: The manipulation condition and weights have to be of same length."

        if not self._interpolate:
            weights = [int(e) for e in weights]

        if self._has_input_transform:
            m = np.eye(3)
            self._generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        w_avg = self._generator.mapping.w_avg
        w_avg = (
            w_avg if len(w_avg.shape) == 1 else w_avg.mean(dim=tuple(range(len(w_avg.shape) - 1)))
        )
        wn_tensors = torch.vstack(candidates.wn_candidates.w_tensors) - w_avg

        """Get w0 vector."""
        w0_weights = (candidates.w0_candidates.weights,)
        assert all(
            (sum(w) == 1 for w in w0_weights)
        ), f"Error: w0 weight do not sum up to one: {w0_weights}."
        w0_weight_tensor = torch.as_tensor(w0_weights, device=self._device)[:, None, None]
        w0_tensors = torch.stack(candidates.w0_candidates.w_tensors) - w_avg
        w0 = (w0_tensors * w0_weight_tensor).sum(dim=0)  # Initialize base using w0 seeds.

        """Here we do style mixing."""
        w = torch.zeros_like(w0, device=self._device, dtype=torch.float32)  # Empty final-w vector
        smw_tensor = torch.as_tensor(weights, device=self._device, dtype=torch.float32)[
            :, None
        ]  # |mix_dims| x 1
        assert (lmd := len(self._mix_dims)) == (
            lsmx := len(cond)
        ), f"Error SMX condition array is not the same size as the mix dimensions ({lmd} vs {lsmx}). This might be due to a mismatch in genome size."
        comp1 = wn_tensors[cond, self._mix_dims, :] * smw_tensor

        w[:, self._mix_dims] += comp1 + w0[:, self._mix_dims] * (1 - smw_tensor)
        w += w_avg

        torch.manual_seed(random_seed)
        img = self.get_image(w)
        return img

    def get_w(self, seed: int, class_idx: int) -> Tensor:
        """
        Generate w vector.

        :param seed: The seed to generate.
        :param class_idx: The label.
        :returns: The w vector.
        """
        torch.manual_seed(seed)

        # Generate latent vectors
        z = torch.randn(size=[1, self._generator.z_dim], device=self._device)
        c = torch.zeros(size=[1, self._generator.c_dim], device=self._device)
        c[:, class_idx] = 1
        w = self._generator.mapping(z=z, c=c, truncation_psi=1, truncation_cutoff=0)
        return w

    def get_image(self, w: Tensor) -> Tensor:
        """
        Get a generated image from the w Vector.

        :param w: The w vector for generation.
        :returns: The generated image.
        """
        out, _ = self._run_synthesis_net(
            self._generator.synthesis, w, noise_mode=self._noise_mode, force_fp32=False
        )
        img = self._transform_image_output(out)
        return img

    @staticmethod
    def _transform_image_output(image: Tensor, full_range: bool = False) -> Tensor:
        """
        Transform generated image output.

        :param image: The image to be transformed.
        :param full_range: If the image should be casted to range [0,255] or [0,1].
        :returns: The transformed image.
        """
        image = image[0].to(torch.float32)  # 1 x C x W x H -> C x W x H
        # Normalize color range.
        image /= image.norm(float("inf"), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
        image = (image * 127.5 + 128).clamp(0, 255) if full_range else ((image + 1) / 2).clamp(0, 1)
        return image

    # ------------------ Copied from https://github.com/NVlabs/stylegan3/blob/main/viz/renderer.py -----------------------
    @staticmethod
    def _run_synthesis_net(net, *args, capture_layer=None, **kwargs):  # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [
                out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]
            ]
            for idx, out in enumerate(outputs):
                if out.ndim == 5:  # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == "":
                    name = "output"
                if len(outputs) > 1:
                    name += f":{idx}"
                if name in unique_names:
                    suffix = 2
                    while f"{name}_{suffix}" in unique_names:
                        suffix += 1
                    name += f"_{suffix}"
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split(".")[-1]
                layers.append(dnnlib.EasyDict(name=name, shape=shape, dtype=dtype))
                if name == capture_layer:
                    raise CaptureSuccess(out)

        hooks = [module.register_forward_hook(module_hook) for module in net.modules()]
        try:
            out = net(*args, **kwargs)
        except CaptureSuccess as e:
            out = e.out
        for hook in hooks:
            hook.remove()
        return out, layers

    @staticmethod
    def _apply_affine_transformation(x, mat, up=4, **filter_kwargs):
        _N, _C, H, W = x.shape
        mat = torch.as_tensor(mat).to(dtype=torch.float32, device=x.device)

        # Construct filter.
        f = _construct_affine_bandlimit_filter(mat, up=up, **filter_kwargs)
        assert f.ndim == 2 and f.shape[0] == f.shape[1] and f.shape[0] % 2 == 1
        p = f.shape[0] // 2

        # Construct sampling grid.
        theta = mat.inverse()
        theta[:2, 2] *= 2
        theta[0, 2] += 1 / up / W
        theta[1, 2] += 1 / up / H
        theta[0, :] *= W / (W + p / up * 2)
        theta[1, :] *= H / (H + p / up * 2)
        theta = theta[:2, :3].unsqueeze(0).repeat([x.shape[0], 1, 1])
        g = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)

        # Resample image.
        y = upfirdn2d.upsample2d(x=x, f=f, up=up, padding=p)
        z = torch.nn.functional.grid_sample(
            y, g, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Form mask.
        m = torch.zeros_like(y)
        c = p * 2 + 1
        m[:, :, c:-c, c:-c] = 1
        m = torch.nn.functional.grid_sample(
            m, g, mode="nearest", padding_mode="zeros", align_corners=False
        )
        return z, m


def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(
        1 - aflt * up
    )
    yi, xi = torch.meshgrid(taps, taps)
    xo, yo = (torch.stack([xi, yi], dim=2) @ mat[:2, :2].t()).unbind(2)

    # Convolution of two oriented 2D sinc filters.
    fi = _sinc(xi * cutoff_in) * _sinc(yi * cutoff_in)
    fo = _sinc(xo * cutoff_out) * _sinc(yo * cutoff_out)
    f = torch.fft.ifftn(torch.fft.fftn(fi) * torch.fft.fftn(fo)).real

    # Convolution of two oriented 2D Lanczos windows.
    wi = _lanczos_window(xi, a) * _lanczos_window(yi, a)
    wo = _lanczos_window(xo, a) * _lanczos_window(yo, a)
    w = torch.fft.ifftn(torch.fft.fftn(wi) * torch.fft.fftn(wo)).real

    # Construct windowed FIR filter.
    f = f * w

    # Finalize.
    c = (aflt - amax) * up
    f = f.roll([aflt * up - 1] * 2, dims=[0, 1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0, 2], keepdim=True) / (up**2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f


def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float("inf"))
    return torch.where(y < 1e-30, torch.ones_like(x), z)


def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))


class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out
