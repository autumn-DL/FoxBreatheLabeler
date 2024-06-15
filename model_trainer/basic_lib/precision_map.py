from typing import Literal, Union

_PRECISION_INPUT_INT = Literal[64, 32, 16]
_PRECISION_INPUT_STR_ALIAS_CONVERSION = {"64": "64-true", "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"}
_PRECISION_INPUT_STR_ALIAS_CONVERSIONB = {"64-true": "64", "32-true": "32", "16-mixed": "16", "bf16-mixed": "bf16"}
_PRECISION_INPUT_STR_ALIAS = Literal["64", "32", "16", "bf16"]
_PRECISION_INPUT_STR = Literal["16-mixed", "bf16-mixed", "32-true", "64-true"]
_PRECISION_INPUT = Union[_PRECISION_INPUT_INT, _PRECISION_INPUT_STR, _PRECISION_INPUT_STR_ALIAS]


def cov_precision(p):
    p = str(p)
    p1 = _PRECISION_INPUT_STR_ALIAS_CONVERSIONB.get(p)
    if p1 is not None:
        p = p1
    return p
