import torch
import numpy as np
from torch import Tensor
from torch_utils.ops import upfirdn2d
import dnnlib
from ._mix_candidate import CandidateList


class StyleMixer:
    """
    A class geared towards style-mixing using style layers in a GAN.

    This class is heavily influenced by the Renderer class in the StyleGAN3 repo.
    """
    _generator: torch.nn.Module
    _device: torch.device
    _has_input_transform: bool

    _pinned_bufs: dict  # Custom buffer for GPU? NVIDIA stuff I guess.
    _mix_dims: Tensor  # Range for dimensions to mix styles with.

    def __init__(
            self,
            generator: torch.nn.Module,
            device: torch.device,
            mix_dims: tuple[int, int]
    ):
        """
        Initialize the StyleMixer object.

        :param generator: The generator network to use.
        :param device: The torch device to use (should be cuda).
        :param mix_dims: The w-dimensions to use for mixing (range index).
        """
        self._generator=generator
        self._device=device
        self._has_input_transform = (hasattr(generator.synthesis, 'input') and hasattr(generator.synthesis.input, 'transform'))
        self._pinned_bufs = {}
        self._mix_dims = torch.arange(*mix_dims, device=self._device)  # Dimensions of w -> we only want to mix style layers.

    def mix(
            self,
            candidates: CandidateList,
            smx_cond: list[int],
            smx_weights: list[float],
            random_seed: int = 0,
            noise_mode: str = "random",
    ) -> Tensor:
        """
        Generate data using style mixing.

        This function is heavily inspired by the Renderer class of the original StyleGANv3 codebase.

        :param candidates: The candidates used for style-mixing.
        :param smx_cond: The style mix conditions (layer combinations).
        :param smx_weights: The weights for mixing layers.
        :param random_seed: The seed for randomization.
        :param noise_mode: The noise to use for style generation (const, random).
        :returns: The generated image (C x H x W).
        """
        assert len(smx_cond) == len(smx_weights), "Error: The parameters have to be of same length."

        if self._has_input_transform:
            m = np.eye(3)
            # TODO: maybe add custom transformations
            self._generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        """Generate latents."""
        num_candidates = len(candidates)

        all_zs = np.zeros([num_candidates, self._generator.z_dim], dtype=np.float32)  # Latent inputs
        all_cs = np.zeros([num_candidates, self._generator.c_dim], dtype=np.float32)  # Input classes
        all_cs[list(range(num_candidates)), candidates.labels] = 1  # Set classes in class vector

        # Custom cast to device by NVIDIA
        all_zs = self._to_device(torch.from_numpy(all_zs))
        all_cs = self._to_device(torch.from_numpy(all_cs))

        ws_average = self._generator.mapping.w_avg
        all_ws = self._generator.mapping(z=all_zs, c=all_cs, truncation_psi=1, truncation_cutoff=0) - ws_average

        """Get candidates for w0 calculation."""
        w0_weights, w0_w_indices = candidates.w0_candidates.weights, candidates.w0_candidates.w_indices
        assert sum(w0_weights) == 1, f"Error: w0 weight do not sum up to one: {w0_weights}."
        weight_tensor = torch.as_tensor(w0_weights, device=self._device)[:, None, None]
        w0 = (all_ws[w0_w_indices] * weight_tensor).sum(dim=0, keepdim=True)  # Initialize base using w0 seeds.
        w = w0.clone().detach()  # Clone w0 for calculation of w -> we want them seperate.
        """
        Here we do style mixing.

        Since we want to mix w0 with wn we take the ws in wn to mix and apply their weights.
        smx indices do not contain w0  --> the index of wn in all_ws is different to index of wn in smx_indices.
        Here we convert the indices to condition such that we know which w to take for each layer (If only one candidate this is array of equal integers).
        """
        wn_w_cond = [candidates.wn_candidates.w_indices[cond] for cond in smx_cond]
        smw_tensor = torch.as_tensor(smx_weights, device=self._device)[:, None]  # |_mix_dims| x 1
        w[self._mix_dims] += all_ws[wn_w_cond, self._mix_dims] * smw_tensor + w0[self._mix_dims] * -(smw_tensor-1)
        w = w / 2 + ws_average

        torch.manual_seed(random_seed)
        out, _ = self._run_synthesis_net(self._generator.synthesis, w[None,:,:], noise_mode=noise_mode, force_fp32=False)

        """Convert the output to an image format."""
        sel = out[0].to(torch.float32)  # 1 x C x W x H -> C x W x H
        img = sel / sel.norm(float('inf'), dim=[1, 2], keepdim=True).clip(1e-8, 1e8)  # Normalize color range.
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Standardize to fit RGB range.
        return img

    # ------------------ Copied from https://github.com/NVlabs/stylegan3/blob/main/viz/renderer.py -----------------------
    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def _to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    @staticmethod
    def _run_synthesis_net(net, *args, capture_layer=None, **kwargs): # => out, layers
        submodule_names = {mod: name for name, mod in net.named_modules()}
        unique_names = set()
        layers = []

        def module_hook(module, _inputs, outputs):
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [out for out in outputs if isinstance(out, torch.Tensor) and out.ndim in [4, 5]]
            for idx, out in enumerate(outputs):
                if out.ndim == 5: # G-CNN => remove group dimension.
                    out = out.mean(2)
                name = submodule_names[module]
                if name == '':
                    name = 'output'
                if len(outputs) > 1:
                    name += f':{idx}'
                if name in unique_names:
                    suffix = 2
                    while f'{name}_{suffix}' in unique_names:
                        suffix += 1
                    name += f'_{suffix}'
                unique_names.add(name)
                shape = [int(x) for x in out.shape]
                dtype = str(out.dtype).split('.')[-1]
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
        z = torch.nn.functional.grid_sample(y, g, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Form mask.
        m = torch.zeros_like(y)
        c = p * 2 + 1
        m[:, :, c:-c, c:-c] = 1
        m = torch.nn.functional.grid_sample(m, g, mode='nearest', padding_mode='zeros', align_corners=False)
        return z, m


def _construct_affine_bandlimit_filter(mat, a=3, amax=16, aflt=64, up=4, cutoff_in=1, cutoff_out=1):
    assert a <= amax < aflt
    mat = torch.as_tensor(mat).to(torch.float32)

    # Construct 2D filter taps in input & output coordinate spaces.
    taps = ((torch.arange(aflt * up * 2 - 1, device=mat.device) + 1) / up - aflt).roll(1 - aflt * up)
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
    f = f.roll([aflt * up - 1] * 2, dims=[0,1])[c:-c, c:-c]
    f = torch.nn.functional.pad(f, [0, 1, 0, 1]).reshape(amax * 2, up, amax * 2, up)
    f = f / f.sum([0,2], keepdim=True) / (up ** 2)
    f = f.reshape(amax * 2 * up, amax * 2 * up)[:-1, :-1]
    return f

def _sinc(x):
    y = (x * np.pi).abs()
    z = torch.sin(y) / y.clamp(1e-30, float('inf'))
    return torch.where(y < 1e-30, torch.ones_like(x), z)

def _lanczos_window(x, a):
    x = x.abs() / a
    return torch.where(x < 1, _sinc(x), torch.zeros_like(x))

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out
