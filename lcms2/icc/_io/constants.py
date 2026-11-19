"""ICC signatures and enumerations."""

TAG_DESC = b"desc"
TAG_CPRT = b"cprt"
TAG_WTPT = b"wtpt"
TAG_BKPT = b"bkpt"
TAG_RXYZ = b"rXYZ"
TAG_GXYZ = b"gXYZ"
TAG_BXYZ = b"bXYZ"
TAG_RTRC = b"rTRC"
TAG_GTRC = b"gTRC"
TAG_BTRC = b"bTRC"
TAG_KTRC = b"kTRC"
TAG_A2B0 = b"A2B0"
TAG_A2B1 = b"A2B1"
TAG_B2A0 = b"B2A0"
TAG_B2A1 = b"B2A1"
TAG_CHAD = b"chad"
TAG_DMND = b"dmnd"
TAG_DMDD = b"dmdd"
TAG_CIIS = b"ciis"

TYPE_CURV = b"curv"
TYPE_PARA = b"para"
TYPE_XYZ = b"XYZ "
TYPE_TEXT = b"text"
TYPE_DESC = b"desc"
TYPE_MLUC = b"mluc"
TYPE_SIG = b"sig "
TYPE_SF32 = b"sf32"
TYPE_UF32 = b"uf32"
TYPE_MFT1 = b"mft1"
TYPE_MFT2 = b"mft2"
TYPE_MAB = b"mAB "
TYPE_MBA = b"mBA "
TYPE_MPET = b"mpet"
TYPE_CHRM = b"chrm"

ELEM_MATF = b"matf"
ELEM_CVST = b"cvst"
ELEM_CLUT = b"clut"
ELEM_BACS = b"bACS"
ELEM_EACS = b"eACS"
ELEM_CALC = b"calc"

CURVE_SEGMENT_FORMULA = 0x0500
CURVE_SEGMENT_SAMPLED = 0x0000

INTENT_PERCEPTUAL = 0
INTENT_RELATIVE_COLORIMETRIC = 1
INTENT_SATURATION = 2
INTENT_ABSOLUTE_COLORIMETRIC = 3

CS_XYZ = b"XYZ "
CS_LAB = b"Lab "
CS_RGB = b"RGB "
CS_GRAY = b"GRAY"
CS_CMYK = b"CMYK"
CS_HSV = b"HSV "
CS_3CLR = b"3CLR"

CHANNELS_BY_CS = {
    CS_XYZ: 3,
    CS_LAB: 3,
    CS_RGB: 3,
    CS_GRAY: 1,
    CS_CMYK: 4,
    CS_HSV: 3,
    CS_3CLR: 3,
}

CLASS_INPUT = b"scnr"
CLASS_DISPLAY = b"mntr"
CLASS_OUTPUT = b"prtr"
CLASS_COLORSPACE = b"spac"
CLASS_ABSTRACT = b"abst"
CLASS_LINK = b"link"
CLASS_NAMED = b"nmcl"

PLATFORM_NONE = b"\x00\x00\x00\x00"

D50_WHITE = (0.9642, 1.0, 0.8249)
