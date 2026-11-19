import os


current_dir = os.path.dirname(__file__)


if __name__ == "__main__":
    fn = os.path.abspath(os.path.join(current_dir, "..", "Little-CMS", "include", "lcms2.h"))
    with open(fn, "r") as fd:
        for line in fd:
            if line.startswith("#define TYPE_"):
                parsed = line.split()
                var_name = parsed[1]
                key = var_name.replace("TYPE_", "")
                print("{" + f"\"{key}\", {var_name}" + "},")