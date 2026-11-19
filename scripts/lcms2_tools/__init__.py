"""Project commands for python-lcms2: ``python scripts/dev.py lcms2 --help``."""

import typer

app = typer.Typer(name="lcms2", help="python-lcms2 maintenance commands.", no_args_is_help=True)


@app.command("lcms-version")
def lcms_version() -> None:
    """Print the LittleCMS version of the vendored sources."""
    from ._paths import LCMS_HEADER

    for line in LCMS_HEADER.read_text(encoding="utf-8").splitlines():
        if line.startswith("#define LCMS_VERSION"):
            v = int(line.split()[2])
            typer.echo(f"{v // 1000}.{v % 1000 // 10}")
            return
    raise typer.Exit(1)
