from . import _lcms2
import numpy as np
import os


class CMSError(Exception):
    def __init__(self, message="LibCMS2 error"):
        super().__init__(message)


def create_profile(profile, white_point=None, degrees=2, temperature=None):
    if profile in ("sRGB", "Lab", "XYZ"):
        coordinates = None
        if profile == "Lab":
            if isinstance(white_point, str):
                coordinates = np.array(WHITE_POINTS[degrees][white_point]["xyY"])
            else:
                if isinstance(temperature, (int, float)):
                    wp_list = _lcms2.white_point_from_temperature(temperature)
                    coordinates = np.array(wp_list)
                else:
                    if white_point is not None:
                        coordinates = np.array(white_point)
                        if coordinates.size != 3:
                            raise CMSError(
                                "white_point should be an array-like object of size 3 containing xyY coordinates of the white point")
        p = _lcms2.create_profile(profile, coordinates)
        return p

    if profile in RGB_COLORSPACES.keys():
        space = RGB_COLORSPACES[profile]
        profile_white_point = space["white point"]
        if profile_white_point not in WHITE_POINTS[degrees].keys():
            raise CMSError(f"Unknown white point `{profile_white_point}`")
        wp = np.array(WHITE_POINTS[degrees][profile_white_point]["xyY"])
        red = np.array(space["red"])
        green = np.array(space["green"])
        blue = np.array(space["blue"])
        tc = space["tone curve"]
        curve_type = int(tc["type"])
        parameters = np.array(tc["parameters"])
        p = _lcms2.create_rgb_profile(wp, red, green, blue, curve_type, parameters)
        return p

    raise CMSError(
        f"Invalid profile '{profile}'. It must be one of built in profiles ('sRGB', 'Lab' or 'XYZ') or supported RGB profiles.")


def open_profile(filename):
    if not isinstance(filename, str):
        raise CMSError("filename must be a string containing a path to a profile")
    if not os.path.isfile(filename):
        raise CMSError(f"Unable to find '{filename}'.")

    p = _lcms2.open_profile(filename)
    return p


def profile_from_memory(buffer):
    if not isinstance(buffer, bytes):
        raise CMSError("filename must be a bytes object")

    p = _lcms2.profile_from_memory(buffer)
    return p


"""
White points of standard illuminants for 2 degrees and 10 degrees field of view
xyY are the coordinates of the white point,
CCT is correlated color temperature in Kelvins
"""
WHITE_POINTS = {2: {'A': {'xyY': [0.44758, 0.40745, 1.0], 'CCT': 2856.0, 'description': 'incandescent / tungsten'},
                    'B': {'xyY': [0.34842, 0.35161, 1.0], 'CCT': 4874.0,
                          'description': 'obsolete, direct sunlight at noon'},
                    'C': {'xyY': [0.31006, 0.31616, 1.0], 'CCT': 6774.0,
                          'description': 'obsolete, average / North sky daylight NTSC 1953, PAL-M'},
                    'D50': {'xyY': [0.34567, 0.3585, 1.0], 'CCT': 5003.0,
                            'description': 'horizon light, ICC profile PCS'},
                    'D55': {'xyY': [0.33242, 0.34743, 1.0], 'CCT': 5503.0,
                            'description': 'mid-morning / mid-afternoon daylight'},
                    'D65': {'xyY': [0.31272, 0.32903, 1.0], 'CCT': 6504.0,
                            'description': 'noon daylight: television, sRGB color space'},
                    'D75': {'xyY': [0.29902, 0.31485, 1.0], 'CCT': 7504.0, 'description': 'North sky daylight'},
                    'D93': {'xyY': [0.28315, 0.29711, 1.0], 'CCT': 9305.0,
                            'description': 'high-efficiency blue phosphor monitors, BT.2035, NTSC-J'},
                    'E': {'xyY': [0.33333, 0.33333, 1.0], 'CCT': 5454.0, 'description': 'equal energy'},
                    'F1': {'xyY': [0.3131, 0.33727, 1.0], 'CCT': 6430.0, 'description': 'daylight fluorescent'},
                    'F2': {'xyY': [0.37208, 0.37529, 1.0], 'CCT': 4230.0, 'description': 'cool white fluorescent'},
                    'F3': {'xyY': [0.4091, 0.3943, 1.0], 'CCT': 3450.0, 'description': 'white fluorescent'},
                    'F4': {'xyY': [0.44018, 0.40329, 1.0], 'CCT': 2940.0, 'description': 'warm white fluorescent'},
                    'F5': {'xyY': [0.31379, 0.34531, 1.0], 'CCT': 6350.0, 'description': 'daylight fluorescent'},
                    'F6': {'xyY': [0.3779, 0.38835, 1.0], 'CCT': 4150.0, 'description': 'light white fluorescent'},
                    'F7': {'xyY': [0.31292, 0.32933, 1.0], 'CCT': 6500.0,
                           'description': 'D65 simulator, daylight simulator'},
                    'F8': {'xyY': [0.34588, 0.35875, 1.0], 'CCT': 5000.0,
                           'description': 'D50 simulator, Sylvania F40 Design 50'},
                    'F9': {'xyY': [0.37417, 0.37281, 1.0], 'CCT': 4150.0,
                           'description': 'cool white deluxe fluorescent'},
                    'F10': {'xyY': [0.34609, 0.35986, 1.0], 'CCT': 5000.0, 'description': 'Philips TL85, Ultralume 50'},
                    'F11': {'xyY': [0.38052, 0.37713, 1.0], 'CCT': 4000.0, 'description': 'Philips TL84, Ultralume 40'},
                    'F12': {'xyY': [0.43695, 0.40441, 1.0], 'CCT': 3000.0, 'description': 'Philips TL83, Ultralume 30'},
                    'FL3.1': {'xyY': [0.4407, 0.4033, 1.0], 'CCT': 2932.0, 'description': 'standard halophosphate'},
                    'FL3.2': {'xyY': [0.3808, 0.3734, 1.0], 'CCT': 3965.0, 'description': 'standard halophosphate'},
                    'FL3.3': {'xyY': [0.3153, 0.3439, 1.0], 'CCT': 6280.0, 'description': 'standard halophosphate'},
                    'FL3.4': {'xyY': [0.4429, 0.4043, 1.0], 'CCT': 2904.0, 'description': 'DeLuxe type'},
                    'FL3.5': {'xyY': [0.3749, 0.3672, 1.0], 'CCT': 4086.0, 'description': 'DeLuxe type'},
                    'FL3.6': {'xyY': [0.3488, 0.36, 1.0], 'CCT': 4894.0, 'description': 'DeLuxe type'},
                    'FL3.7': {'xyY': [0.4384, 0.4045, 1.0], 'CCT': 2979.0, 'description': 'Three-band'},
                    'FL3.8': {'xyY': [0.382, 0.3832, 1.0], 'CCT': 4006.0, 'description': 'Three-band'},
                    'FL3.9': {'xyY': [0.3499, 0.3591, 1.0], 'CCT': 4853.0, 'description': 'Three-band'},
                    'FL3.10': {'xyY': [0.3455, 0.356, 1.0], 'CCT': 5000.0, 'description': 'Three-band'},
                    'FL3.11': {'xyY': [0.3245, 0.3434, 1.0], 'CCT': 5854.0, 'description': 'Three-band'},
                    'FL3.12': {'xyY': [0.4377, 0.4037, 1.0], 'CCT': 2984.0, 'description': 'Multi-band'},
                    'FL3.13': {'xyY': [0.383, 0.3724, 1.0], 'CCT': 3896.0, 'description': 'Multi-band'},
                    'FL3.14': {'xyY': [0.3447, 0.3609, 1.0], 'CCT': 5045.0, 'description': 'Multi-band'},
                    'FL3.15': {'xyY': [0.3127, 0.3288, 1.0], 'CCT': 6509.0, 'description': 'D65 Simulator'},
                    'HP1': {'xyY': [0.533, 0.415, 1.0], 'CCT': 1959.0,
                            'description': 'Standard high pressure sodium lamp'},
                    'HP2': {'xyY': [0.4778, 0.4158, 1.0], 'CCT': 2506.0,
                            'description': 'Color-enhanced high-pressure sodium lamp'},
                    'HP3': {'xyY': [0.4302, 0.4075, 1.0], 'CCT': 3144.0,
                            'description': 'High pressure metal halide lamp'},
                    'HP4': {'xyY': [0.3812, 0.3797, 1.0], 'CCT': 4002.0,
                            'description': 'High pressure metal halide lamp'},
                    'HP5': {'xyY': [0.3776, 0.3713, 1.0], 'CCT': 4039.0,
                            'description': 'High pressure metal halide lamp'},
                    'LED-B1': {'xyY': [0.456, 0.4078, 1.0], 'CCT': 2733.0, 'description': 'phosphor-converted blue'},
                    'LED-B2': {'xyY': [0.4357, 0.4012, 1.0], 'CCT': 2998.0, 'description': 'phosphor-converted blue'},
                    'LED-B3': {'xyY': [0.3756, 0.3723, 1.0], 'CCT': 4103.0, 'description': 'phosphor-converted blue'},
                    'LED-B4': {'xyY': [0.3422, 0.3502, 1.0], 'CCT': 5109.0, 'description': 'phosphor-converted blue'},
                    'LED-B5': {'xyY': [0.3118, 0.3236, 1.0], 'CCT': 6598.0, 'description': 'phosphor-converted blue'},
                    'LED-BH1': {'xyY': [0.4474, 0.4066, 1.0], 'CCT': 2851.0,
                                'description': 'mixing of phosphor-converted blue LED and red LED (blue-hybrid)'},
                    'LED-RGB1': {'xyY': [0.4557, 0.4211, 1.0], 'CCT': 2840.0,
                                 'description': 'mixing of red, green, and blue LEDs'},
                    'LED-V1': {'xyY': [0.4548, 0.4044, 1.0], 'CCT': 2724.0, 'description': 'phosphor-converted violet'},
                    'LED-V2': {'xyY': [0.3781, 0.3775, 1.0], 'CCT': 4070.0, 'description': 'phosphor-converted violet'},
                    'ID50': {'xyY': [0.3432, 0.3602, 1.0], 'CCT': 5098.0, 'description': 'natural indoor daylight'},
                    'ID65': {'xyY': [0.3107, 0.3307, 1.0], 'CCT': 6603.0, 'description': 'natural indoor daylight'}},
                10: {'A': {'xyY': [0.45117, 0.40594, 1.0], 'CCT': 2856.0, 'description': 'incandescent / tungsten'},
                     'B': {'xyY': [0.3498, 0.3527, 1.0], 'CCT': 4874.0,
                           'description': 'obsolete, direct sunlight at noon'},
                     'C': {'xyY': [0.31039, 0.31905, 1.0], 'CCT': 6774.0,
                           'description': 'obsolete, average / North sky daylight NTSC 1953, PAL-M'},
                     'D50': {'xyY': [0.34773, 0.35952, 1.0], 'CCT': 5003.0,
                             'description': 'horizon light, ICC profile PCS'},
                     'D55': {'xyY': [0.33411, 0.34877, 1.0], 'CCT': 5503.0,
                             'description': 'mid-morning / mid-afternoon daylight'},
                     'D65': {'xyY': [0.31382, 0.331, 1.0], 'CCT': 6504.0,
                             'description': 'noon daylight: television, sRGB color space'},
                     'D75': {'xyY': [0.29968, 0.3174, 1.0], 'CCT': 7504.0, 'description': 'North sky daylight'},
                     'D93': {'xyY': [0.28327, 0.30043, 1.0], 'CCT': 9305.0,
                             'description': 'high-efficiency blue phosphor monitors, BT.2035, NTSC-J'},
                     'E': {'xyY': [0.33333, 0.33333, 1.0], 'CCT': 5454.0, 'description': 'equal energy'},
                     'F1': {'xyY': [0.31811, 0.33559, 1.0], 'CCT': 6430.0, 'description': 'daylight fluorescent'},
                     'F2': {'xyY': [0.37925, 0.36733, 1.0], 'CCT': 4230.0, 'description': 'cool white fluorescent'},
                     'F3': {'xyY': [0.41761, 0.38324, 1.0], 'CCT': 3450.0, 'description': 'white fluorescent'},
                     'F4': {'xyY': [0.4492, 0.39074, 1.0], 'CCT': 2940.0, 'description': 'warm white fluorescent'},
                     'F5': {'xyY': [0.31975, 0.34246, 1.0], 'CCT': 6350.0, 'description': 'daylight fluorescent'},
                     'F6': {'xyY': [0.3866, 0.37847, 1.0], 'CCT': 4150.0, 'description': 'light white fluorescent'},
                     'F7': {'xyY': [0.31569, 0.3296, 1.0], 'CCT': 6500.0,
                            'description': 'D65 simulator, daylight simulator'},
                     'F8': {'xyY': [0.34902, 0.35939, 1.0], 'CCT': 5000.0,
                            'description': 'D50 simulator, Sylvania F40 Design 50'},
                     'F9': {'xyY': [0.37829, 0.37045, 1.0], 'CCT': 4150.0,
                            'description': 'cool white deluxe fluorescent'},
                     'F10': {'xyY': [0.3509, 0.35444, 1.0], 'CCT': 5000.0, 'description': 'Philips TL85, Ultralume 50'},
                     'F11': {'xyY': [0.38541, 0.37123, 1.0], 'CCT': 4000.0,
                             'description': 'Philips TL84, Ultralume 40'},
                     'F12': {'xyY': [0.44256, 0.39717, 1.0], 'CCT': 3000.0,
                             'description': 'Philips TL83, Ultralume 30'}}}

RGB_COLORSPACES = {'Adobe RGB (1998)': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D65',
                                        'red': [0.64, 0.33, 0.297361], 'green': [0.21, 0.71, 0.627355],
                                        'blue': [0.15, 0.06, 0.075285]},
                   'Apple RGB': {'tone curve': {'type': 1, 'parameters': [1.8]}, 'white point': 'D65',
                                 'red': [0.625, 0.34, 0.244634], 'green': [0.28, 0.595, 0.672034],
                                 'blue': [0.155, 0.07, 0.083332]},
                   'Best RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D50',
                                'red': [0.7347, 0.2653, 0.228457], 'green': [0.215, 0.775, 0.737352],
                                'blue': [0.13, 0.035, 0.034191]},
                   'Beta RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D50',
                                'red': [0.6888, 0.3112, 0.303273], 'green': [0.1986, 0.7551, 0.663786],
                                'blue': [0.1265, 0.0352, 0.032941]},
                   'Bruce RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D65',
                                 'red': [0.64, 0.33, 0.240995], 'green': [0.28, 0.65, 0.683554],
                                 'blue': [0.15, 0.06, 0.075452]},
                   'CIE RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'E',
                               'red': [0.735, 0.265, 0.176204], 'green': [0.274, 0.717, 0.812985],
                               'blue': [0.167, 0.009, 0.010811]},
                   'ColorMatch RGB': {'tone curve': {'type': 1, 'parameters': [1.8]}, 'white point': 'D50',
                                      'red': [0.63, 0.34, 0.274884], 'green': [0.295, 0.605, 0.658132],
                                      'blue': [0.15, 0.075, 0.066985]},
                   'Don RGB 4': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D50',
                                 'red': [0.696, 0.3, 0.27835], 'green': [0.215, 0.765, 0.68797],
                                 'blue': [0.13, 0.035, 0.03368]},
                   'ECI RGB v2': {
                       'tone curve': {'type': 4, 'parameters': [3.000000, 0.862076, 0.137924, 0.110703, 0.080002]},
                       'white point': 'D50', 'red': [0.67, 0.33, 0.32025],
                       'green': [0.21, 0.71, 0.602071], 'blue': [0.14, 0.08, 0.077679]},
                   'Ekta Space PS5': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D50',
                                      'red': [0.695, 0.305, 0.260629], 'green': [0.26, 0.7, 0.734946],
                                      'blue': [0.11, 0.005, 0.004425]},
                   'NTSC RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'C',
                                'red': [0.67, 0.33, 0.298839], 'green': [0.21, 0.71, 0.586811],
                                'blue': [0.14, 0.08, 0.11435]},
                   'PAL/SECAM RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D65',
                                     'red': [0.64, 0.33, 0.222021], 'green': [0.29, 0.6, 0.706645],
                                     'blue': [0.15, 0.06, 0.071334]},
                   'ProPhoto RGB': {'tone curve': {'type': 1, 'parameters': [1.8]}, 'white point': 'D50',
                                    'red': [0.7347, 0.2653, 0.28804], 'green': [0.1596, 0.8404, 0.711874],
                                    'blue': [0.0366, 0.0001, 8.6e-05]},
                   'SMPTE-C RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D65',
                                   'red': [0.63, 0.34, 0.212395], 'green': [0.31, 0.595, 0.701049],
                                   'blue': [0.155, 0.07, 0.086556]},
                   '_sRGB': {'tone curve': {'type': 4, 'parameters': [2.399994, 0.947861, 0.052139, 0.077393, 0.040451]},
                            'white point': 'D65', 'red': [0.64, 0.33, 0.212656],
                            'green': [0.3, 0.6, 0.715158], 'blue': [0.15, 0.06, 0.072186]},
                   'Wide Gamut RGB': {'tone curve': {'type': 1, 'parameters': [2.2]}, 'white point': 'D50',
                                      'red': [0.735, 0.265, 0.258187], 'green': [0.115, 0.826, 0.724938],
                                      'blue': [0.157, 0.018, 0.016875]}}
