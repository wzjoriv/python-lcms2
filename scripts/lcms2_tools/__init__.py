"""Project commands for python-lcms2: ``python scripts/dev.py lcms2 --help``."""

import re
import subprocess

import typer

app = typer.Typer(name="lcms2", help="python-lcms2 maintenance commands.", no_args_is_help=True)

STABLE_TAG = re.compile(r"^lcms2\.(\d+)(?:\.(\d+))?$")


@app.command("lcms-version")
def lcms_version() -> None:
    """Print the LittleCMS version of the vendored sources."""
    from ._header import lcms_version as parse
    from ._paths import LCMS_HEADER

    typer.echo(parse(LCMS_HEADER.read_text(encoding="utf-8")))


@app.command("formats")
def formats() -> None:
    """Regenerate lcms2/_ext/formats_table.h from the vendored lcms2.h."""
    from ._header import TABLE, write_table
    from ._paths import LCMS_HEADER, ROOT

    n = write_table(LCMS_HEADER)
    typer.echo(f"wrote {n} formats to {TABLE.relative_to(ROOT)}")


def _git(*args: str) -> str:
    from ._paths import LCMS

    out = subprocess.run(["git", "-C", str(LCMS), *args], check=True, capture_output=True, text=True)
    return out.stdout


def newest_stable_tag(tags: list[str]) -> str | None:
    """The highest ``lcms2.X`` / ``lcms2.X.Y`` tag; release candidates are ignored."""
    stable = []
    for tag in tags:
        m = STABLE_TAG.match(tag.strip())
        if m:
            stable.append(((int(m.group(1)), int(m.group(2) or 0)), tag.strip()))
    return max(stable)[1] if stable else None


@app.command("update")
def update(
    tag: str = typer.Option("", help="Check out this tag instead of the newest stable one."),
    fetch: bool = typer.Option(True, help="Fetch tags from the LittleCMS remote first."),
) -> None:
    """Move vendor/Little-CMS to the newest stable release and regenerate the format table.

    Prints the CMSAPI functions added or removed between the old and new lcms2.h, so new
    API can be bound and removed API dropped.
    """
    from ._header import TABLE, cmsapi_functions, lcms_version, write_table
    from ._paths import LCMS_HEADER, ROOT

    if fetch:
        _git("fetch", "--tags", "--quiet")
    target = tag or newest_stable_tag(_git("tag", "--list", "lcms2.*").splitlines())
    if not target:
        typer.echo("no stable lcms2.X[.Y] tag found", err=True)
        raise typer.Exit(1)

    old_text = LCMS_HEADER.read_text(encoding="utf-8")
    _git("checkout", "--quiet", target)
    new_text = LCMS_HEADER.read_text(encoding="utf-8")
    typer.echo(f"LittleCMS {lcms_version(old_text)} -> {lcms_version(new_text)} ({target})")

    n = write_table(LCMS_HEADER)
    typer.echo(f"wrote {n} formats to {TABLE.relative_to(ROOT)}")

    old, new = cmsapi_functions(old_text), cmsapi_functions(new_text)
    for label, names in (("added", new - old), ("removed", old - new)):
        typer.echo(f"{label} CMSAPI functions: {len(names)}")
        for name in sorted(names):
            typer.echo(f"  {name}")
    typer.echo("Commit the submodule bump and lcms2/_ext/formats_table.h together.")
