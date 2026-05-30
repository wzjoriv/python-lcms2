import sys
import os
from setuptools import setup, Extension
import platform
import numpy as np


def get_architecture():
    num_bits, _ = platform.architecture()
    machine = platform.machine()
    if  num_bits.startswith("64"):
        arch = "arm64" if machine.lower().startswith("arm") else "x64"
        return arch
    else:
        arch = "arm" if machine.lower().startswith("arm") else "x86"
        return arch


def get_additional_paths():
    include_path = list()
    libraries_path = list()
    if sys.platform.startswith("win"):
        import winreg
        r = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        key_name = "SOFTWARE\\WOW6432Node\\Microsoft\\Microsoft SDKs\\Windows"
        try:
            key = winreg.OpenKey(r, key_name)
            num_keys, num_values, timestamp = winreg.QueryInfoKey(key)
            recent_timestamp = 0
            root_path = ""
            version = ""
            for n in range(num_keys):
                subkey_name = winreg.EnumKey(key, n)
                subkey = winreg.OpenKey(r, key_name + "\\" + subkey_name)
                nk, nv, timestamp = winreg.QueryInfoKey(subkey)
                if timestamp > recent_timestamp:
                    record = dict()
                    for q in range(nv):
                        name, value, _ = winreg.EnumValue(subkey, q)
                        record[name] = value
                    root_path = record["InstallationFolder"]
                    version = record["ProductVersion"] + ".0"
            for p in ("cppwinrt", "shared", "ucrt", "um", "winrt"):
                folder = os.path.sep.join((root_path + "Include", version, p))
                include_path.append(folder)
            arch = get_architecture()
            for p in ("ucrt", "um"):
                folder = os.path.sep.join((root_path + "Lib", version, p, arch))
                libraries_path.append(folder)
        except:
            raise RuntimeError("Unable to locate Windows SDK on this platform. Is it installed?")

    return include_path, libraries_path


SOURCES = ["src/lcms2/_lcms2.c",
           "Little-CMS/src/cmsalpha.c",
           "Little-CMS/src/cmscam02.c",
           "Little-CMS/src/cmscgats.c",
           "Little-CMS/src/cmscnvrt.c",
           "Little-CMS/src/cmserr.c",
           "Little-CMS/src/cmsgamma.c",
           "Little-CMS/src/cmsgmt.c",
           "Little-CMS/src/cmshalf.c",
           "Little-CMS/src/cmsintrp.c",
           "Little-CMS/src/cmsio0.c",
           "Little-CMS/src/cmsio1.c",
           "Little-CMS/src/cmslut.c",
           "Little-CMS/src/cmsmd5.c",
           "Little-CMS/src/cmsmtrx.c",
           "Little-CMS/src/cmsnamed.c",
           "Little-CMS/src/cmsopt.c",
           "Little-CMS/src/cmspack.c",
           "Little-CMS/src/cmspcs.c",
           "Little-CMS/src/cmsplugin.c",
           "Little-CMS/src/cmsps2.c",
           "Little-CMS/src/cmssamp.c",
           "Little-CMS/src/cmssm.c",
           "Little-CMS/src/cmstypes.c",
           "Little-CMS/src/cmsvirt.c",
           "Little-CMS/src/cmswtpnt.c",
           "Little-CMS/src/cmsxform.c"]

INCLUDE_DIRECTORIES = ["Little-CMS/include", "Little-CMS/src", np.get_include()]
LIBRARY_DIRECTORIES = list()

extra_include, extra_libs = get_additional_paths()

setup_args = dict(
    ext_modules=[
        Extension(
            name="lcms2._lcms2",
            language="c",
            define_macros=[('MAJOR_VERSION', '0'), ('MINOR_VERSION', '4')],
            include_dirs=INCLUDE_DIRECTORIES + extra_include,
            library_dirs=LIBRARY_DIRECTORIES + extra_libs,
            sources=SOURCES,
            py_limited_api=True
        )
    ]
)

setup(**setup_args)