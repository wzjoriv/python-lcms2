"""ICC profile container: 128-byte header, tag table and raw tag data."""

from __future__ import annotations

import datetime as _dt
import hashlib
import struct
from dataclasses import dataclass

from .._errors import ICCParseError
from . import binary as bn

HEADER_SIZE = 128


@dataclass
class ICCHeader:
    """Decoded ICC profile header fields."""

    size: int
    cmm: bytes
    version_major: int
    version_minor: int
    version_bugfix: int
    device_class: bytes
    data_colour_space: bytes
    pcs: bytes
    created: _dt.datetime
    primary_platform: bytes
    flags: int
    manufacturer: bytes
    model: bytes
    attributes: int
    rendering_intent: int
    pcs_illuminant: tuple
    creator: bytes
    profile_id: bytes

    @property
    def version(self) -> tuple:
        """``(major, minor, bugfix)``."""
        return (self.version_major, self.version_minor, self.version_bugfix)

    @property
    def version_tuple_str(self) -> str:
        """Version as ``'major.minor.bugfix'``."""
        return f"{self.version_major}.{self.version_minor}.{self.version_bugfix}"


class ICCProfile:
    """A parsed ICC profile.

    Attributes
    ----------
    header : ICCHeader
        Decoded header.
    tags : dict of bytes to bytes
        Raw tag data keyed by 4-byte tag signature.
    raw : bytes
        The complete profile bytes.
    """

    def __init__(self, header: ICCHeader, tags: dict[bytes, bytes], raw: bytes):
        self.header = header
        self.tags = tags
        self.raw = raw

    @classmethod
    def load(cls, path_or_bytes) -> ICCProfile:
        """Load an ICC profile from a file path or raw bytes.

        Parameters
        ----------
        path_or_bytes : str or bytes
            Path to an ICC file, or its raw file contents.

        Returns
        -------
        ICCProfile
            The parsed profile.
        """
        if isinstance(path_or_bytes, (bytes, bytearray)):
            data = bytes(path_or_bytes)
        else:
            with open(path_or_bytes, "rb") as f:
                data = f.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> ICCProfile:
        """Parse a raw ICC file byte stream into header + tag table.

        Parameters
        ----------
        data : bytes
            Full contents of an ICC profile file.

        Returns
        -------
        ICCProfile
            The parsed profile.

        Raises
        ------
        ICCParseError
            If the data is too small, has a bad file signature, an
            unsupported major version, or a truncated/out-of-bounds tag
            table.
        """
        if len(data) < HEADER_SIZE + 4:
            raise ICCParseError(f"file too small to be an ICC profile ({len(data)} bytes)")

        size = bn.read_u32(data, 0)
        cmm = bn.sig(data, 4)
        version_word = bn.read_u32(data, 8)
        version_major = (version_word >> 24) & 0xFF
        version_minor = (version_word >> 20) & 0xF
        version_bugfix = (version_word >> 16) & 0xF
        device_class = bn.sig(data, 12)
        data_colour_space = bn.sig(data, 16)
        pcs = bn.sig(data, 20)
        created = bn.read_dateTime(data, 24)
        acsp = bn.sig(data, 36)
        if acsp != b"acsp":
            raise ICCParseError(
                f"bad profile file signature: expected b'acsp', got {acsp!r} "
                "(not a valid ICC profile, or the file is corrupt)"
            )
        primary_platform = bn.sig(data, 40)
        flags = bn.read_u32(data, 44)
        manufacturer = bn.sig(data, 48)
        model = bn.sig(data, 52)
        attributes = bn.read_u64(data, 56)
        rendering_intent = bn.read_u32(data, 64)
        pcs_illuminant = bn.read_XYZNumber(data, 68)
        creator = bn.sig(data, 80)
        profile_id = bytes(data[84:100])

        if version_major not in (2, 3, 4, 5):
            raise ICCParseError(f"unsupported/unknown major ICC version byte: {version_major}")

        header = ICCHeader(
            size=size,
            cmm=cmm,
            version_major=version_major,
            version_minor=version_minor,
            version_bugfix=version_bugfix,
            device_class=device_class,
            data_colour_space=data_colour_space,
            pcs=pcs,
            created=created,
            primary_platform=primary_platform,
            flags=flags,
            manufacturer=manufacturer,
            model=model,
            attributes=attributes,
            rendering_intent=rendering_intent,
            pcs_illuminant=pcs_illuminant,
            creator=creator,
            profile_id=profile_id,
        )

        tag_count = bn.read_u32(data, HEADER_SIZE)
        table_off = HEADER_SIZE + 4
        needed = table_off + tag_count * 12
        if needed > len(data):
            raise ICCParseError(f"truncated tag table: need {needed} bytes, have {len(data)}")

        tags: dict[bytes, bytes] = {}
        for i in range(tag_count):
            entry_off = table_off + i * 12
            tsig = bn.sig(data, entry_off)
            toff = bn.read_u32(data, entry_off + 4)
            tsize = bn.read_u32(data, entry_off + 8)
            if toff < HEADER_SIZE or toff + tsize > len(data):
                raise ICCParseError(
                    f"tag {tsig!r} out of bounds: offset={toff} size={tsize} file_len={len(data)}"
                )
            tags[tsig] = bytes(data[toff : toff + tsize])

        return cls(header, tags, data)

    def get_tag(self, sig4: bytes) -> bytes | None:
        """Return the raw bytes of a tag, or None if not present.

        Parameters
        ----------
        sig4 : bytes
            4-byte tag signature, e.g. ``b'wtpt'``.

        Returns
        -------
        bytes or None
        """
        return self.tags.get(sig4)

    def has_tag(self, sig4: bytes) -> bool:
        """Return whether the profile contains a tag with this signature.

        Parameters
        ----------
        sig4 : bytes
            4-byte tag signature.

        Returns
        -------
        bool
        """
        return sig4 in self.tags

    def __repr__(self):
        h = self.header
        return (
            f"<ICCProfile v{h.version_tuple_str} class={h.device_class!r} "
            f"space={h.data_colour_space!r} pcs={h.pcs!r} tags={len(self.tags)}>"
        )

    @staticmethod
    def build_bytes(
        *,
        version=(4, 4, 0),
        device_class=b"spac",
        data_colour_space=b"RGB ",
        pcs=b"XYZ ",
        rendering_intent=1,
        pcs_illuminant=(0.9642, 1.0, 0.8249),
        cmm=b"\x00\x00\x00\x00",
        primary_platform=b"\x00\x00\x00\x00",
        manufacturer=b"\x00\x00\x00\x00",
        model=b"\x00\x00\x00\x00",
        attributes=0,
        flags=0,
        creator=b"lcms",
        created: _dt.datetime | None = None,
        tags: dict[bytes, bytes] = None,
    ) -> bytes:
        """Serialize a profile from header fields and raw tag data.

        Identical tag payloads share one data block, every block is padded
        to 4 bytes, and v4+ profiles carry the MD5 profile ID of ICC.1.

        Returns
        -------
        bytes
        """
        tags = tags or {}
        created = created or _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)

        tag_items = list(tags.items())
        n = len(tag_items)
        table_size = 4 + n * 12
        data_start = HEADER_SIZE + table_size

        placed: dict[bytes, tuple] = {}
        body = bytearray()
        entries = []
        cursor = data_start
        for sig4, payload in tag_items:
            if payload in placed:
                off, size = placed[payload]
            else:
                off = cursor
                size = len(payload)
                body += payload
                pad = bn.pad4(size)
                body += b"\x00" * pad
                cursor += size + pad
                placed[payload] = (off, size)
            entries.append((sig4, off, size))

        total_size = data_start + len(body)

        version_word = (
            (version[0] & 0xFF) << 24 | (version[1] & 0xF) << 20 | (version[2] & 0xF) << 16
        )

        header = bytearray(HEADER_SIZE)
        struct.pack_into(">I", header, 0, total_size)
        header[4:8] = cmm.ljust(4, b"\x00")[:4]
        struct.pack_into(">I", header, 8, version_word)
        header[12:16] = device_class.ljust(4, b"\x00")[:4]
        header[16:20] = data_colour_space.ljust(4, b"\x00")[:4]
        header[20:24] = pcs.ljust(4, b"\x00")[:4]
        header[24:36] = bn.write_dateTime(created)
        header[36:40] = b"acsp"
        header[40:44] = primary_platform.ljust(4, b"\x00")[:4]
        struct.pack_into(">I", header, 44, flags)
        header[48:52] = manufacturer.ljust(4, b"\x00")[:4]
        header[52:56] = model.ljust(4, b"\x00")[:4]
        struct.pack_into(">Q", header, 56, attributes)
        struct.pack_into(">I", header, 64, rendering_intent)
        header[68:80] = bn.write_XYZNumber(pcs_illuminant)
        header[80:84] = creator.ljust(4, b"\x00")[:4]
        header[84:100] = b"\x00" * 16

        table = struct.pack(">I", n)
        for sig4, off, size in entries:
            table += sig4.ljust(4, b"\x00")[:4] + struct.pack(">II", off, size)

        full = bytes(header) + table + bytes(body)

        if version[0] < 4:
            return full
        hashable = bytearray(full)
        struct.pack_into(">I", hashable, 44, 0)
        struct.pack_into(">I", hashable, 64, 0)
        hashable[84:100] = b"\x00" * 16
        with_id = bytearray(full)
        with_id[84:100] = hashlib.md5(bytes(hashable)).digest()
        return bytes(with_id)
