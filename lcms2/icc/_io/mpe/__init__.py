"""Encoders, decoders and evaluators for the ICC.2 multiProcessElementsType ('mpet').

Elements: matrixElement ('matf'), curveSetElement ('cvst') of segmented
curves ('curf') with formula ('parf') and sampled ('samf') segments,
CLUTElement ('clut') and calculatorElement ('calc').
"""

from .calc import decode_calc_element, encode_calc_element, validate_calc_element
from .calc_eval import eval_calc_element
from .container import decode_element, decode_mpet, encode_mpet, validate_mpet
from .curves import (
    constant_segment,
    decode_curveset_element,
    encode_curveset_element,
    encode_segmented_curve,
    eval_curveset_channel,
    eval_formula,
)
from .elements import (
    decode_clut_element,
    decode_matrix_element,
    encode_clut_element,
    encode_matrix_element,
)

__all__ = [
    "constant_segment",
    "decode_calc_element",
    "decode_clut_element",
    "decode_curveset_element",
    "decode_element",
    "decode_matrix_element",
    "decode_mpet",
    "encode_calc_element",
    "encode_clut_element",
    "encode_curveset_element",
    "encode_matrix_element",
    "encode_mpet",
    "encode_segmented_curve",
    "eval_calc_element",
    "eval_curveset_channel",
    "eval_formula",
    "validate_calc_element",
    "validate_mpet",
]
