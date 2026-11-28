"""
Reading and writing CGATS.17 / IT8 measurement files (targets, characterisation data).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

from . import _lcms2

__all__ = ["CGATS"]


class _Properties(MutableMapping):
    """Dict-like view of the header properties of the current table."""

    def __init__(self, owner: CGATS):
        self._owner = owner

    def __getitem__(self, key: str) -> str:
        h = self._owner._sel()
        value = _lcms2.it8_get_property(h, str(key))
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: str | float) -> None:
        h = self._owner._sel()
        if isinstance(value, bool):
            value = int(value)
        quoted = isinstance(value, str)
        text = value if quoted else repr(float(value)) if isinstance(value, float) else str(value)
        _lcms2.it8_set_property(h, str(key), text, quoted)

    def __delitem__(self, key: str) -> None:
        raise TypeError("CGATS properties cannot be removed; build a new CGATS without the property")

    def __iter__(self) -> Iterator[str]:
        return iter(_lcms2.it8_properties(self._owner._sel()))

    def __len__(self) -> int:
        return len(_lcms2.it8_properties(self._owner._sel()))

    def __repr__(self) -> str:
        return repr(dict(self))


class CGATS:
    """A CGATS.17 / IT8.7 file: one or more tables of named columns and patch rows.

    All accessors act on the *current table*; change it with :meth:`select` or the ``table``
    property. Create one with ``CGATS()`` (empty), :meth:`load`, :meth:`from_bytes` or
    :meth:`from_columns`.

    See Also
    --------
    CGATS.from_columns : Build a file from a mapping of columns.
    CGATS.load : Read a file.

    Examples
    --------
    >>> import numpy as np
    >>> from lcms2.cgats import CGATS
    >>> it8 = CGATS.from_columns({"SAMPLE_ID": ["A1", "A2"], "LAB_L": [10.0, 20.0]})
    >>> it8.columns
    ['SAMPLE_ID', 'LAB_L']
    >>> it8.column("LAB_L").tolist()
    [10.0, 20.0]
    >>> CGATS.from_bytes(it8.to_bytes())["A2", "LAB_L"]
    20.0
    """

    def __init__(self):
        self._handle = _lcms2.it8_new()
        self._table = 0
        self.properties = _Properties(self)

    # ---- Construction / persistence ----
    @classmethod
    def from_bytes(cls, data: bytes) -> CGATS:
        """Parse CGATS text.

        Parameters
        ----------
        data : bytes
            The CGATS file contents.

        Returns
        -------
        CGATS
            The parsed file.

        Raises
        ------
        CMSError
            If the data is not valid CGATS (the message names the offending line).

        See Also
        --------
        CGATS.to_bytes : The inverse.

        Examples
        --------
        >>> text = CGATS.from_columns({"A": [1.0, 2.0]}).to_bytes()
        >>> CGATS.from_bytes(text).column("A").tolist()
        [1.0, 2.0]
        """
        self = cls.__new__(cls)
        self._handle = _lcms2.it8_load(bytes(data))
        self._table = 0
        self.properties = _Properties(self)
        return self

    @classmethod
    def load(cls, path: str | os.PathLike) -> CGATS:
        """Read a CGATS file from ``path`` (the file is read in Python and then closed).

        Parameters
        ----------
        path : str or os.PathLike
            Path of the CGATS file.

        Returns
        -------
        CGATS
            The parsed file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        CMSError
            If the file is not valid CGATS.

        See Also
        --------
        CGATS.save : Write a file.

        Examples
        --------
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "t.txt")
        >>> CGATS.from_columns({"A": [1.0]}).save(path)
        >>> CGATS.load(path).columns
        ['A']
        """
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())

    @classmethod
    def from_columns(
        cls,
        data: Mapping[str, Sequence[Any]],
        sheet_type: str | None = None,
        properties: Mapping[str, str | float] | None = None,
    ) -> CGATS:
        """Build a single-table CGATS from a mapping of column name to values.

        Parameters
        ----------
        data : mapping of str to sequence
            Columns of equal length; numbers are stored as numbers, anything else as text.
        sheet_type : str, optional
            Sheet type (the first line of the file). Defaults to ``"CGATS.17"``.
        properties : mapping, optional
            Header properties to set. Defaults to None.

        Returns
        -------
        CGATS
            The new file.

        Raises
        ------
        ValueError
            If there are no columns or the columns differ in length.

        See Also
        --------
        CGATS.data : The columns of the current table.

        Examples
        --------
        >>> CGATS.from_columns({"SAMPLE_ID": ["A1"], "X": [0.5]}).n_rows
        1
        """
        names = list(data)
        cols = [list(np.asarray(data[n]).tolist()) for n in names]
        rows = {len(c) for c in cols}
        if not names or len(rows) != 1:
            raise ValueError("data must have at least one column and all columns must have the same length")
        self = cls()
        if sheet_type is not None:
            self.sheet_type = sheet_type
        for key, value in (properties or {}).items():
            self.properties[key] = value
        self.define_table(names, rows.pop())
        for c, column in enumerate(cols):
            for r, value in enumerate(column):
                self._set_cell(r, c, value)
        return self

    def to_bytes(self) -> bytes:
        """Serialise every table to CGATS text.

        Returns
        -------
        bytes
            The CGATS file contents.

        See Also
        --------
        CGATS.from_bytes : The inverse.

        Examples
        --------
        >>> CGATS.from_columns({"A": [1.0]}).to_bytes().startswith(b"CGATS.17")
        True
        """
        out = _lcms2.it8_save(self._handle)
        return out

    def save(self, path: str | os.PathLike) -> None:
        """Write the file to ``path``.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.

        See Also
        --------
        CGATS.load : Read a file.

        Examples
        --------
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "t.txt")
        >>> CGATS.from_columns({"A": [1.0]}).save(path)
        >>> os.path.getsize(path) > 0
        True
        """
        data = self.to_bytes()
        with open(path, "wb") as fh:
            fh.write(data)

    # ---- Tables ----
    def _sel(self) -> Any:
        _lcms2.it8_set_table(self._handle, self._table)
        return self._handle

    def __len__(self) -> int:
        return _lcms2.it8_table_count(self._handle)

    @property
    def table(self) -> int:
        """Index of the current table."""
        return self._table

    @table.setter
    def table(self, n: int) -> None:
        self.select(n)

    def select(self, n: int) -> CGATS:
        """Make table ``n`` current and return ``self``.

        Parameters
        ----------
        n : int
            Table index.

        Returns
        -------
        CGATS
            This object, for chaining.

        Raises
        ------
        IndexError
            If ``n`` is not a table of the file (``0 <= n < len(self)``).

        See Also
        --------
        CGATS.add_table : Append and select a new table.

        Examples
        --------
        >>> CGATS.from_columns({"A": [1.0]}).select(0).table
        0
        """
        n = int(n)
        count = len(self)
        if not 0 <= n < count:
            raise IndexError(f"table {n} out of range; the file has {count} table(s) (0..{count - 1})")
        self._table = n
        return self

    def add_table(self) -> int:
        """Append an empty table, make it current and return its index.

        Returns
        -------
        int
            Index of the new table.

        See Also
        --------
        CGATS.define_table : Declare its columns.

        Examples
        --------
        >>> CGATS.from_columns({"A": [1.0]}).add_table()
        1
        """
        n = len(self)
        _lcms2.it8_set_table(self._handle, n)
        self._table = n
        return n

    @property
    def sheet_type(self) -> str | None:
        """Sheet type of the current table (for example ``"CGATS.17"`` or ``"IT8.7/2"``)."""
        return _lcms2.it8_sheet_type(self._sel())

    @sheet_type.setter
    def sheet_type(self, value: str) -> None:
        _lcms2.it8_set_sheet_type(self._sel(), str(value))

    # ---- Columns and rows ----
    def define_table(self, columns: Iterable[str], n_rows: int) -> None:
        """Declare the column names and row count of the (empty) current table.

        Parameters
        ----------
        columns : iterable of str
            Column names.
        n_rows : int
            Number of patches.

        Raises
        ------
        CMSError
            If the table already has columns; call :meth:`add_table` first.

        See Also
        --------
        CGATS.add_table : Append a new table first.

        Examples
        --------
        >>> it8 = CGATS()
        >>> it8.define_table(["SAMPLE_ID", "X"], 2)
        >>> it8.n_rows
        2
        """
        _lcms2.it8_define_table(self._sel(), [str(c) for c in columns], int(n_rows))

    @property
    def columns(self) -> list[str]:
        """Column names (the DATA_FORMAT) of the current table."""
        return _lcms2.it8_data_format(self._sel())

    @property
    def n_rows(self) -> int:
        """Number of patches (rows) of the current table."""
        return _lcms2.it8_dims(self._sel())[0]

    @property
    def patches(self) -> list[str]:
        """Patch names, from the ``SAMPLE_ID`` column (or the first column when absent)."""
        cols = self.columns
        if not cols:
            return []
        idx = next((i for i, c in enumerate(cols) if c.upper() == "SAMPLE_ID"), 0)
        return [str(self._get_cell(r, idx)) for r in range(self.n_rows)]

    def _col_index(self, column: str | int) -> int:
        cols = self.columns
        if isinstance(column, (int, np.integer)):
            if not 0 <= column < len(cols):
                raise IndexError(f"column {column} out of range; the table has {len(cols)} column(s)")
            return int(column)
        for i, name in enumerate(cols):
            if name.lower() == str(column).lower():
                return i
        raise KeyError(f"no column named {column!r}; available columns: {', '.join(cols)}")

    def _row_index(self, patch: str | int) -> int:
        if isinstance(patch, (int, np.integer)):
            if not 0 <= patch < self.n_rows:
                raise IndexError(f"row {patch} out of range; the table has {self.n_rows} row(s)")
            return int(patch)
        names = self.patches
        try:
            return names.index(str(patch))
        except ValueError:
            raise KeyError(f"no patch named {patch!r}; first patches: {', '.join(names[:5])}") from None

    def _get_cell(self, row: int, col: int) -> str | None:
        return _lcms2.it8_get_cell(self._sel(), row, col)

    def _set_cell(self, row: int, col: int, value: Any) -> None:
        if isinstance(value, (float, np.floating)):
            text = repr(float(value))
        elif isinstance(value, (bool, int, np.integer)):
            text = str(int(value))
        else:
            text = str(value)
        _lcms2.it8_set_cell(self._sel(), row, col, text)

    def get(self, patch: str | int, column: str | int) -> float | str | None:
        """Value of one cell: a ``float`` when it is numeric, otherwise a string (``None`` if empty).

        Parameters
        ----------
        patch : str or int
            Patch name (``SAMPLE_ID``) or row index.
        column : str or int
            Column name (case-insensitive) or index.

        Returns
        -------
        float, str or None
            The cell value.

        Raises
        ------
        KeyError
            For an unknown patch or column name.
        IndexError
            For a row or column index out of range.

        See Also
        --------
        CGATS.set : Change a cell.

        Examples
        --------
        >>> CGATS.from_columns({"SAMPLE_ID": ["A1"], "X": [0.5]}).get("A1", "x")
        0.5
        """
        text = self._get_cell(self._row_index(patch), self._col_index(column))
        return _parse(text)

    def set(self, patch: str | int, column: str | int, value: str | float) -> None:
        """Set one cell from a number or string; see :meth:`get` for the addressing.

        Parameters
        ----------
        patch : str or int
            Patch name (``SAMPLE_ID``) or row index.
        column : str or int
            Column name (case-insensitive) or index.
        value : str or float
            The new value.

        Raises
        ------
        KeyError
            For an unknown patch or column name.
        IndexError
            For a row or column index out of range.

        See Also
        --------
        CGATS.get : Read a cell.

        Examples
        --------
        >>> it8 = CGATS.from_columns({"SAMPLE_ID": ["A1"], "X": [0.5]})
        >>> it8.set("A1", "X", 0.75)
        >>> it8.get(0, 1)
        0.75
        """
        self._set_cell(self._row_index(patch), self._col_index(column), value)

    def __getitem__(self, key: tuple[str | int, str | int]) -> float | str | None:
        patch, column = key
        return self.get(patch, column)

    def __setitem__(self, key: tuple[str | int, str | int], value: str | float) -> None:
        patch, column = key
        self.set(patch, column, value)

    def column(self, name: str | int) -> np.ndarray:
        """One column: a ``float64`` array if every cell is numeric, else an array of strings.

        Empty numeric cells become ``nan``.

        Parameters
        ----------
        name : str or int
            Column name (case-insensitive) or index.

        Returns
        -------
        numpy.ndarray
            The column values.

        Raises
        ------
        KeyError
            For an unknown column name.
        IndexError
            For a column index out of range.

        See Also
        --------
        CGATS.data : All columns at once.

        Examples
        --------
        >>> CGATS.from_columns({"X": [0.5, 1.5]}).column("X").tolist()
        [0.5, 1.5]
        """
        c = self._col_index(name)
        cells = [self._get_cell(r, c) for r in range(self.n_rows)]
        numbers = [_parse(t) for t in cells]
        if all(isinstance(v, float) or v is None for v in numbers):
            return np.array([np.nan if v is None else v for v in numbers], dtype=np.float64)
        return np.array(["" if t is None else t for t in cells], dtype=str)

    @property
    def data(self) -> dict[str, np.ndarray]:
        """All columns of the current table as ``{name: array}`` (see :meth:`column`)."""
        return {name: self.column(i) for i, name in enumerate(self.columns)}

    def to_dict(self) -> dict:
        """Plain-Python snapshot of the whole file (JSON friendly).

        Returns
        -------
        dict
            ``{"tables": [{"sheet_type", "properties", "columns": {name: list}}, ...]}``.

        See Also
        --------
        CGATS.data : The columns of the current table as arrays.

        Examples
        --------
        >>> CGATS.from_columns({"X": [0.5]}).to_dict()["tables"][0]["columns"]
        {'X': [0.5]}
        """
        keep = self._table
        tables = []
        try:
            for n in range(len(self)):
                self.select(n)
                tables.append(
                    {
                        "sheet_type": self.sheet_type,
                        "properties": dict(self.properties),
                        "columns": {k: v.tolist() for k, v in self.data.items()},
                    }
                )
        finally:
            self._table = keep
        return {"tables": tables}

    def __repr__(self) -> str:
        return (
            f"CGATS(tables={len(self)}, current={self._table}, sheet_type={self.sheet_type!r}, "
            f"columns={len(self.columns)}, rows={self.n_rows})"
        )


def _parse(text: str | None) -> float | str | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text
