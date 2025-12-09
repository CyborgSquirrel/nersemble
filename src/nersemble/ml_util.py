import abc
import dataclasses as dc
import logging
import os
import os.path as osp
import re
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, ClassVar, Optional, Self

import humanize
import imageio as iio
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image

logger = logging.getLogger(__name__)

try:
  import unimatch_utils.frame_utils
except ModuleNotFoundError:
  logger.warning("Unimatch not setup, can't import it...")

RE_VIDEO_FRAME = re.compile(r"(?P<name>.*[.]mp4):(?P<index>\d+)$")

def load_img(
  img: str | Path,
  *,
  handle_video=True,
  output_type=torch.Tensor,
) -> torch.Tensor:
  if isinstance(img, str):
    img = Path(img)
  if handle_video:
    if isinstance(img, Path):
      m = RE_VIDEO_FRAME.match(img.name)
      if m is not None:
        img = img.with_name(m.group("name"))
        index = int(m.group("index"))
        with iio.imopen(img, "r", plugin="pyav") as f:
          img = f.read(index=index)
  if isinstance(img, Path):
    img = Image.open(img)
    img = img.convert("RGB")
    img = np.asarray(img)

  if output_type is np.ndarray:
    return img

  img = torch.from_numpy(img).float()
  return img

def unimatch_load_img(img: str | Path | Image.Image | np.ndarray) -> torch.Tensor:
  if isinstance(img, Path):
    img = str(img)
  if isinstance(img, str):
    img = unimatch_utils.frame_utils.read_gen(img)
  if isinstance(img, Image.Image):
    img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
  if isinstance(img, np.ndarray):
    dtype = np.uint8
    if img.dtype != dtype:
      raise TypeError(f"Expected {dtype=}, got {img.dtype=}")
  img = img[..., :3]  # [H, W, C]
  img = torch.from_numpy(img).permute(2, 0, 1).float()  # [C, H, W]
  return img

def unimatch_get_transform(
  imgs: list[torch.Tensor],  # [C, H, W]
  *,
  padding_factor=8,
  flow_size=None,
  flow_scale=None,
) -> tuple[
  Callable[[torch.Tensor], torch.Tensor],
  Callable[[torch.Tensor], torch.Tensor],
]:
  orig_shape = tuple(imgs[0].shape)
  orig_size = orig_shape[-2:]
  if not all(img.shape == orig_shape for img in imgs):
    raise RuntimeError("Shapes of all images must be equal")

  # Resize to nearest size or specified size
  if flow_size is None:
    size = orig_size

    # Scale size
    if flow_scale is not None:
      size = (
        size[-2] * flow_scale,
        size[-1] * flow_scale,
      )

    # Snap size to padding factor
    size = (
      int(np.ceil(size[-2] / padding_factor)) * padding_factor,
      int(np.ceil(size[-1] / padding_factor)) * padding_factor,
    )

    flow_size = size
  assert isinstance(flow_size, tuple)

  # The model is trained with size: width > height
  inference_transpose = False
  if orig_shape[-2] > orig_shape[-1]:
    inference_transpose = True

  def trans_fn(img: torch.Tensor):
    # Resize
    if flow_size != orig_size:
      img = img.unsqueeze(0)
      img = F.interpolate(
        img,
        size=flow_size,
        mode="bilinear",
        align_corners=True,
      )
      img = img.squeeze(0)

    # Transpose
    if inference_transpose:
      img = torch.transpose(img, -2, -1)

    return img

  def restore_fn(img: torch.Tensor, *, scale_flow: bool):
    # Transpose
    if inference_transpose:
      img = torch.transpose(img, -2, -1)

    # Resize
    if flow_size != orig_size:
      img = img.unsqueeze(0)
      img = F.interpolate(
        img,
        size=orig_size,
        mode="bilinear",
        align_corners=True,
      )
      img = img.squeeze(0)

      # NOTE(andrei): I _think_ this adapts the flow to the new size, but
      # honestly not 100% sure...
      if scale_flow:
        img[:, 0] = img[:, 0] * orig_size[-1] / flow_size[-1]
        img[:, 1] = img[:, 1] * orig_size[-2] / flow_size[-2]

    return img

  return trans_fn, restore_fn

def flow_to_image_rgb(
  # [H, W, 2]
  flow: torch.Tensor | np.ndarray,
) -> Image.Image:
  shape = tuple(flow.shape)
  match shape:
    case (_h, _w, 2): pass
    case _: raise TypeError(f"Unexpected {shape=}")

  if isinstance(flow, torch.Tensor):
    flow = flow.cpu().numpy()

  thresh = np.max(np.abs(flow), axis=(0, 1))

  lo = -thresh
  hi =  thresh

  flow = (flow - lo) / (hi - lo)
  flow = flow * 128 + 128
  flow = np.stack([
    flow[..., 0],
    np.zeros(shape[:-1]),
    flow[..., 1],
  ], axis=-1)

  flow = np.clip(flow, 0, 255)
  flow = flow.astype(np.uint8)

  flow = Image.fromarray(flow)
  return flow

def hsv_to_rgb(
  # [..., 3]
  hsv: np.ndarray,
) -> np.ndarray:
  h = hsv[..., 0]
  s = hsv[..., 1]
  v = hsv[..., 2]

  h6 = h * 6.0
  i = np.floor(h6).astype(np.int8)
  f = h6 - i

  p = v * (1 - s)
  q = v * (1 - f * s)
  t = v * (1 - (1 - f) * s)

  i_mod = (i % 6)[..., None]  # [..., 1]

  # Prebuild the 6 possible RGB triplets
  candidates = np.stack([
    np.stack([v, t, p], axis=-1),
    np.stack([q, v, p], axis=-1),
    np.stack([p, v, t], axis=-1),
    np.stack([p, q, v], axis=-1),
    np.stack([t, p, v], axis=-1),
    np.stack([v, p, q], axis=-1),
  ], axis=-2)   # [..., 6, 3]

  # Pick the correct one for each pixel
  rgb = np.take_along_axis(candidates, i_mod[..., None], axis=-2)[..., 0, :]

  return rgb

def rgb_to_hsv(
  # [..., 3]
  rgb: np.ndarray
) -> np.ndarray:
  r = rgb[..., 0]
  g = rgb[..., 1]
  b = rgb[..., 2]

  cmax = np.maximum.reduce([r, g, b])
  cmin = np.minimum.reduce([r, g, b])
  delta = cmax - cmin

  # Hue calculation
  h = np.zeros_like(cmax)
  mask = delta != 0

  r_eq_max = (cmax == r) & mask
  g_eq_max = (cmax == g) & mask
  b_eq_max = (cmax == b) & mask

  h[r_eq_max] = ( (g[r_eq_max] - b[r_eq_max]) / delta[r_eq_max] ) % 6
  h[g_eq_max] = ( (b[g_eq_max] - r[g_eq_max]) / delta[g_eq_max] ) + 2
  h[b_eq_max] = ( (r[b_eq_max] - g[b_eq_max]) / delta[b_eq_max] ) + 4

  h = h / 6.0
  h[h < 0] += 1.0  # Ensure hue is in [0, 1]

  # Saturation calculation
  s = np.zeros_like(cmax)
  s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

  # Value calculation
  v = cmax

  hsv = np.stack([h, s, v], axis=-1)
  return hsv

def test_rgb_hsv_conversion():
  space = np.linspace(0, 1, 50)
  rgb = np.meshgrid(space, space, space)
  rgb = np.stack(rgb, axis=-1)
  hsv = rgb_to_hsv(rgb)
  rgb_roundtrip = hsv_to_rgb(hsv)
  assert np.all(np.isclose(rgb, rgb_roundtrip))

# Taken from here https://github.com/lizhihao6/Forward-Warp
def forward_warp(im0, flow, interpolation_mode):
  im0 = im0.to(torch.float32)
  im1 = torch.zeros_like(im0)
  B = im0.shape[0]
  H = im0.shape[2]
  W = im0.shape[3]
  if interpolation_mode == 0:
    for b in range(B):
      for h in range(H):
        for w in range(W):
          x = w + flow[b, h, w, 0]
          y = h + flow[b, h, w, 1]
          nw = (int(torch.floor(x)), int(torch.floor(y)))
          ne = (nw[0]+1, nw[1])
          sw = (nw[0], nw[1]+1)
          se = (nw[0]+1, nw[1]+1)
          p = im0[b, :, h, w]

          if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
            nw_k = (se[0]-x)*(se[1]-y)
            ne_k = (x-sw[0])*(sw[1]-y)
            sw_k = (ne[0]-x)*(y-ne[1])
            se_k = (x-nw[0])*(y-nw[1])
            im1[b, :, nw[1], nw[0]] += nw_k*p
            im1[b, :, ne[1], ne[0]] += ne_k*p
            im1[b, :, sw[1], sw[0]] += sw_k*p
            im1[b, :, se[1], se[0]] += se_k*p
  else:
    round_flow = torch.round(flow)
    for b in range(B):
      for h in range(H):
        for w in range(W):
          x = w + int(round_flow[b, h, w, 0])
          y = h + int(round_flow[b, h, w, 1])
          if x >= 0 and x < W and y >= 0 and y < H:
            im1[b, :, y, x] = im0[b, :, h, w]
  im1 = torch.clip(im1, 0, 255)
  return im1

def numpify(fn):
  """Wrap a function expecting torch arrays to auto-convert inputs to torch and
  outputs to numpy."""

  def conv(value):
    if isinstance(value, np.ndarray):
      print(value.shape)
      return torch.from_numpy(value)
    return value

  def wrapped(*args, **kwargs):
    args = [ conv(val) for val in args ]
    kwargs = { conv(key): conv(val) for key, val in kwargs.items() }
    ret = fn(*args, **kwargs)
    if isinstance(ret, torch.Tensor):
      ret = ret.cpu().numpy()
    return ret

  return wrapped

@dc.dataclass
class MetaExif(abc.ABC):
  @abc.abstractmethod
  def exif_encode(self) -> bytes:
    ...
  @classmethod
  @abc.abstractmethod
  def exif_decode(cls, src: bytes) -> Self:
    ...

  def img_write(self, img: Image.Image):
    exif = img.getexif()
    exif[ExifTags.IFD.MakerNote] = self.exif_encode()
  @classmethod
  def img_read(cls, img: Image.Image) -> Self:
    exif = img.getexif()
    exif_data = exif[ExifTags.IFD.MakerNote]
    return cls.exif_decode(exif_data)

############################################################
#                      HSV Flow Image                      #
############################################################

@dc.dataclass
class FlowExif(MetaExif):
  mag_max: float

  STRUCT_FMT: ClassVar = ">d"
  def exif_encode(self) -> bytes:
    return struct.pack(self.STRUCT_FMT, self.mag_max)
  @classmethod
  def exif_decode(cls, src: bytes) -> Self:
    mag_max, = struct.unpack(cls.STRUCT_FMT, src)
    return cls(mag_max)

def flow_to_hue_flow(
  # [H, W, 2]
  flow: torch.Tensor | np.ndarray,
  *,
  mag_max: Optional[float] = None,
):
  if isinstance(flow, torch.Tensor):
    flow = flow.cpu().numpy()

  shape = tuple(flow.shape)
  if shape[-1] != 2:
    raise TypeError(f"Unexpected {shape=}")

  ang = np.atan2(flow[..., 1], flow[..., 0])
  ang = (ang + np.pi) / (np.pi * 2)

  mag = np.linalg.vector_norm(flow, axis=-1, ord=2)
  if mag_max is None:
    mag_max = np.max(mag).item()
  mag = mag / mag_max
  mag = np.clip(mag, 0, 1)

  flow = np.stack([
    ang,
    mag,
    np.ones(shape[:-1]),
  ], axis=-1)
  flow = hsv_to_rgb(flow)

  return mag_max, flow

def flow_to_image_hue(
  # [H, W, 2]
  flow: torch.Tensor | np.ndarray,
  *,
  exif: bool = True,
  mag_max: Optional[float] = None,
) -> Image.Image:
  with_exif = exif
  del exif

  mag_max, flow = flow_to_hue_flow(flow, mag_max=mag_max)
  flow = np.clip(flow * 255, 0, 255).astype(np.uint8)
  flow = Image.fromarray(flow)

  if with_exif:
    flow_exif = FlowExif(mag_max)
    flow_exif.img_write(flow)

  return flow

# [H, W, 2]
def image_to_flow_hue(
  img: str | Path | Image.Image
) -> np.ndarray:
  if isinstance(img, str):
    img = Path(img)
  if isinstance(img, Path):
    img = Image.open(img)
  if isinstance(img, Image.Image):
    exif = FlowExif.img_read(img)

    img: np.ndarray
    img = np.asarray(img)
    img = img.astype(np.float64)
    img = img / 255

    img = rgb_to_hsv(img)
    ang = img[..., 0] * np.pi * 2 - np.pi
    mag = img[..., 1] * exif.mag_max

    flow = np.stack([np.cos(ang), np.sin(ang)], axis=-1) * mag[..., None]
    return flow
  raise RuntimeError(f"Unexpected {type(img)=}")

def test_image_flow_hue_conversion():
  def test_case(*, rad, mag, rtol=None, atol=None):
    mag_max = mag
    del mag

    space = np.linspace(-1, 1, num=rad)
    x, y = np.meshgrid(space, space)

    flow = np.stack([x, y], axis=-1)
    mag = np.linalg.vector_norm(flow, axis=-1)
    flow[mag > 1, :] = 0

    flow = flow * mag_max

    img = flow_to_image_hue(flow, exif=True)
    flow_roundtrip = image_to_flow_hue(img)

    kwargs = dict(rtol=rtol, atol=atol)
    if kwargs["rtol"] is None: kwargs.pop("rtol")
    if kwargs["atol"] is None: kwargs.pop("atol")

    assert np.all(np.isclose(flow, flow_roundtrip, **kwargs))

  test_case(rad=100, mag=5, atol=0.05)
  test_case(rad=100, mag=100, atol=0.75)

############################################################
#                       Depth Image                        #
############################################################

def depth_to_image(
  path: str | Path,
  # [H, W] | [H, W, 1]
  depth: torch.Tensor | np.ndarray,
):
  shape = tuple(depth.shape)
  match shape:
    case (_h, _w): pass
    case (_h, _w, 1):
      depth = depth[..., 0]
    case _: raise TypeError(f"Unexpected {shape=}")

  if isinstance(depth, torch.Tensor):
    depth = depth.cpu().numpy()

  if isinstance(path, str):
    path = Path(path)
  if path.suffix != ".tiff":
    raise RuntimeError(f"Unsupported {path.suffix=}")

  # NOTE: TIFF supports 16-bit ints, so we bitcast to that.
  tifffile.imwrite(
    path,
    depth.astype(np.float16).view(np.uint16),
    compression="ZLIB",
    photometric="minisblack",
  )

# [H, W, 1]
def image_to_depth(
  path: str | Path,
) -> np.ndarray:
  img = tifffile.imread(path)
  if img.dtype != np.uint16:
    raise RuntimeError(f"Unexpected {img.dtype=}")
  img = img.view(np.float16)
  return img

def test_image_depth_conversion():
  with TemporaryDirectory() as tmp_dir:
    img_path = osp.join(tmp_dir, "depth.tiff")

    depth = np.linspace(0, 50, num=800*400).reshape(800, 400)
    depth_to_image(img_path, depth)
    logger.info("Image size %r", humanize.naturalsize(os.stat(img_path).st_size))
    depth_roundtrip = image_to_depth(img_path)

  assert np.all(np.isclose(depth, depth_roundtrip, rtol=0.001))

############################################################
#                    Deformation Image                     #
############################################################

def deformation_to_image(
  path: str | Path,
  # [H, W, 3]
  deformation: torch.Tensor | np.ndarray,
):
  shape = tuple(deformation.shape)
  match shape:
    case (_h, _w, 3):
      pass
    case _:
      raise TypeError(f"Unexpected {shape=}")

  if isinstance(deformation, torch.Tensor):
    deformation = deformation.cpu().numpy()

  if isinstance(path, str):
    path = Path(path)
  if path.suffix != ".tiff":
    raise RuntimeError(f"Unsupported {path.suffix=}")

  # NOTE: TIFF supports 16-bit ints, so we bitcast to that.
  tifffile.imwrite(
    path,
    deformation.astype(np.float16).view(np.uint16),
    compression="ZLIB",
    photometric="rgb",
  )

# [H, W, 3]
def image_to_deformation(
  path: str | Path,
) -> np.ndarray:
  img = tifffile.imread(path)

  # sanity checks
  dtype = img.dtype
  if dtype != np.uint16:
    raise RuntimeError(f"Unexpected {dtype=}")
  img = img.view(np.float16)

  shape = img.shape
  match shape:
    case (_h, _w, 3):
      pass
    case _:
      raise RuntimeError(f"Unexpected {shape=}")

  return img

def test_image_deformation_conversion():
  with TemporaryDirectory() as tmp_dir:
    img_path = osp.join(tmp_dir, "deformation.tiff")

    deformation = np.linspace(0, 50, num=800*400*3).reshape(800, 400, 3)
    deformation_to_image(img_path, deformation)
    logger.info("Image size %r", humanize.naturalsize(os.stat(img_path).st_size))
    deformation_roundtrip = image_to_deformation(img_path)

  assert np.all(np.isclose(deformation, deformation_roundtrip, rtol=0.001))
