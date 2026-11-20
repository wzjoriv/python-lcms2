"""Low-level LittleCMS2 bindings (lcms2._lcms2). Use the lcms2 package instead."""

from typing import Any

from numpy.typing import NDArray

class LCMSError(ValueError): ...

LCMS_VERSION: int
FORMATS: dict[str, tuple[int, str, int, int, bool]]

Handle = Any

def get_version() -> int:
    """Encoded LittleCMS version of the linked library (2200 for 2.20)."""

# formats.c
def format_info(fmt: int) -> tuple[str, int, int, bool]:
    """Decode a pixel format: (sample dtype, channels including extra, extra channels, planar)."""

def format_for_profile(profile: Any, bytes: int, is_float: bool, extra: int = 0, pcs: bool = False) -> int:
    """Pixel format for the colour space (or PCS) of a profile with `bytes` per sample (0 = double)
    and `extra` extra (alpha) channels.
    """

# profile.c / profile_info.c
def profile_open(data: bytes | bytearray | memoryview) -> Handle:
    """Open an ICC profile from bytes-like data."""

def profile_to_bytes(profile: Any) -> bytes:
    """Serialise a profile."""

def profile_create_srgb() -> Handle:
    """Built-in sRGB."""

def profile_create_xyz() -> Handle:
    """XYZ identity profile (D50 PCS)."""

def profile_create_null() -> Handle:
    """Output profile that discards everything (gray, all zeros)."""

def profile_create_lab(version: int = 4, white_point: Any = None) -> Handle:
    """Lab identity profile; white_point is xyY (D50 when None)."""

def profile_create_rgb(white_point: Any, primaries: Any, curves: Any) -> Handle:
    """Matrix-shaper RGB profile. primaries is (red, green, blue) xyY; curves holds 1 or 3 items,
    each a gamma number, a (type, params) parametric tuple or a ToneCurve.
    """

def profile_create_gray(white_point: Any, curve: Any) -> Handle:
    """Gray profile from a white point and a curve."""

def profile_create_bchsw(
    lut_points: int = 17,
    brightness: float = 0.0,
    contrast: float = 1.0,
    hue: float = 0.0,
    saturation: float = 0.0,
    temp_src: int = 0,
    temp_dst: int = 0,
) -> Handle:
    """Abstract Lab profile adjusting brightness, contrast, hue, saturation and white point."""

def profile_create_ink_limiting(color_space: str | int, limit: float) -> Handle:
    """Ink-limiting device link (CMYK only)."""

def profile_create_linearization(color_space: str | int, curves: Any) -> Handle:
    """Device link applying one curve per channel."""

def profile_get_header(profile: Any) -> dict[str, Any]:
    """Header fields of a profile."""

def profile_set_header(profile: Any, **fields: Any) -> None:
    """Change header fields."""

def profile_compute_id(profile: Any) -> bytes:
    """Compute the MD5 profile ID, store it in the header, return it."""

def profile_get_info(profile: Any, info: int, language: str | None = None, country: str | None = None) -> str:
    """Text of info 0=description, 1=manufacturer, 2=model, 3=copyright."""

def profile_set_info(profile: Any, info: int, text: str, language: str = "en", country: str = "US") -> None:
    """Write an info text tag (0=description, 1=manufacturer, 2=model, 3=copyright)."""

def profile_tags(profile: Any) -> list[str]:
    """Tag signatures."""

def profile_is_tag(profile: Any, sig: str | int) -> bool:
    """Whether the profile has a tag."""

def profile_read_raw_tag(profile: Any, sig: str | int) -> bytes:
    """Serialised tag data (with the type header)."""

def profile_write_raw_tag(profile: Any, sig: str | int, data: bytes) -> None:
    """Store raw tag data."""

def profile_read_xyz(profile: Any, sig: str | int) -> tuple[float, float, float] | None:
    """Read an XYZ tag (wtpt, bkpt, rXYZ, gXYZ, bXYZ, lumi)."""

def profile_read_chad(profile: Any) -> tuple[tuple[float, float, float], ...] | None:
    """Chromatic adaptation matrix (3x3 nested tuples)."""

def profile_read_trc(profile: Any, sig: str | int) -> Handle | None:
    """Copy of a TRC curve (rTRC, gTRC, bTRC, kTRC)."""

def profile_is_matrix_shaper(profile: Any) -> bool:
    """Whether the profile is a matrix-shaper."""

def profile_is_clut(profile: Any, intent: int, direction: int) -> bool:
    """Whether the intent is implemented by a CLUT."""

def profile_is_intent_supported(profile: Any, intent: int, direction: int) -> bool:
    """Direction 0 = input, 1 = output, 2 = proof."""

def get_supported_intents() -> list[tuple[int, str]]:
    """Intents the engine implements."""

def profile_detect_black_point(
    profile: Any, intent: int, flags: int = 0, destination: bool = False
) -> tuple[float, float, float] | None:
    """Black point XYZ, or None when it cannot be detected."""

def profile_detect_tac(profile: Any) -> float:
    """Total area coverage in percent (0 when unknown)."""

def profile_named_colors(
    profile: Any,
) -> list[tuple[str, str, str, tuple[int, int, int], tuple[int, ...]]] | None:
    """Named colours as (name, prefix, suffix, pcs, colorants), or None without a ncl2 tag."""

# transform.c
def transform_create(src: Any, dst: Any, in_fmt: int, out_fmt: int, intent: int, flags: int) -> Handle:
    """Create a transform."""

def transform_create_proofing(
    src: Any, dst: Any, proof: Any, in_fmt: int, out_fmt: int, intent: int, proof_intent: int, flags: int
) -> Handle:
    """Create a soft-proofing transform."""

def transform_create_multi(profiles: Any, in_fmt: int, out_fmt: int, intent: int, flags: int) -> Handle:
    """Create a transform through a chain of profiles."""

def transform_create_extended(
    profiles: Any,
    bpc: Any,
    intents: Any,
    adaptation: Any,
    gamut: Any,
    gamut_position: int,
    in_fmt: int,
    out_fmt: int,
    flags: int,
) -> Handle:
    """Create a transform with per-profile intents, BPC and adaptation states."""

def transform_apply(transform: Any, src: NDArray[Any], dst: NDArray[Any]) -> None:
    """Transform contiguous `src` into `dst` (dtypes and sizes checked against the formats)."""

def transform_formats(transform: Any) -> tuple[int, int]:
    """Input and output pixel formats."""

def transform_to_device_link(transform: Any, version: float, flags: int) -> Handle:
    """Device link profile of a transform."""

def get_alarm_codes() -> tuple[int, ...]:
    """Gamut-check alarm codes (16 values, global)."""

def set_alarm_codes(codes: Any) -> None:
    """Set the gamut-check alarm codes (global)."""

def set_adaptation_state(state: float) -> float:
    """Set the global adaptation state; returns the previous one (negative = query only)."""

# Tone curves
def curve_gamma(gamma: float) -> Handle:
    """Build a pure power-law tone curve Y = X ** gamma."""

def curve_parametric(type: int, params: Any) -> Handle:
    """Build a parametric tone curve (types 1-8, 108, 109; negative for the inverse). The parameter count is validated per type."""

def curve_tabulated(values: Any) -> Handle:
    """Build a tone curve from a 1-D uint16 or float table (floats are on the 0..1 range)."""

def curve_copy(curve: Any) -> Handle:
    """Duplicate the curve."""

def curve_reverse(curve: Any, n: int = 4096) -> Handle:
    """Reverse the curve, sampling n points when no analytic inverse exists."""

def curve_join(first: Any, second: Any, n: int = 4096) -> Handle:
    """Join curves: x -> second^-1(first(x))."""

def curve_smooth(curve: Any, lam: float) -> Handle:
    """Return a smoothed copy of the curve (lam is the smoothing parameter)."""

def curve_eval(curve: Any, x: Any) -> NDArray[Any]:
    """Evaluate the curve over a float array (0..1); returns float64."""

def curve_eval16(curve: Any, x: Any) -> NDArray[Any]:
    """Evaluate the curve over a uint16 array; returns uint16."""

def curve_table(curve: Any) -> NDArray[Any]:
    """The 16-bit table of the curve as a uint16 array."""

def curve_info(curve: Any) -> dict[str, Any]:
    """Flags and sizes: linear, monotonic, descending, multisegment, parametric_type, entries."""

def curve_estimate_gamma(curve: Any, precision: float = 0.01) -> float:
    """Estimate the gamma of the curve."""

# Colorimetry
def delta_e(
    lab1: Any, lab2: Any, method: str, p0: float = 1.0, p1: float = 1.0, p2: float = 1.0
) -> NDArray[Any]:
    """Color difference over two equally-shaped (..., 3) Lab arrays. method: 0=76, 1=94, 2=2000 (p0..p2 = Kl, Kc, Kh), 3=CMC (p0, p1 = l, c), 4=BFD."""

def cam02(
    values: Any, white: Any, yb: float, la: float, surround: int, d: float, reverse: bool
) -> NDArray[Any]:
    """CIECAM02 forward (XYZ to JCh) or reverse (JCh to XYZ) over (..., 3) arrays; XYZ on the 0..100 scale, d=-1 computes the degree of adaptation."""

def white_from_temp(kelvin: Any) -> NDArray[Any]:
    """xyY of the daylight white point (4000-25000 K); output has a trailing axis of 3."""

def temp_from_white(xyy: Any) -> NDArray[Any]:
    """Correlated color temperature in kelvin of (..., 3) xyY values."""

# CGATS / IT8
def it8_new() -> Handle:
    """Create an empty CGATS.17 container."""

def it8_load(data: bytes) -> Handle:
    """Parse CGATS/IT8 text from a bytes-like object."""

def it8_save(it8: Any) -> bytes:
    """Serialise every table to CGATS text."""

def it8_table_count(it8: Any) -> int:
    """Number of tables."""

def it8_set_table(it8: Any, n: int) -> None:
    """Select table n (n equal to the count appends a new table)."""

def it8_sheet_type(it8: Any) -> str | None:
    """The sheet type of the current table."""

def it8_set_sheet_type(it8: Any, sheet_type: str) -> None:
    """Set the sheet type of the current table."""

def it8_properties(it8: Any) -> list[str]:
    """Property names of the current table."""

def it8_get_property(it8: Any, name: str) -> str | None:
    """Value of a property of the current table."""

def it8_set_property(it8: Any, name: str, value: Any, quoted: bool) -> None:
    """Set a property of the current table."""

def it8_data_format(it8: Any) -> list[str]:
    """Column names of the current table."""

def it8_dims(it8: Any) -> tuple[int, int]:
    """(rows, columns) of the current table."""

def it8_define_table(it8: Any, names: Any, rows: int) -> None:
    """Declare columns and row count of the current table."""

def it8_get_cell(it8: Any, row: int, col: int) -> str | None:
    """Cell text, or None when empty."""

def it8_set_cell(it8: Any, row: int, col: int, value: Any) -> None:
    """Set a cell from text."""
