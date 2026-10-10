"""Byte-level RFB server→client gate: the clipboard reaches only the lease holder.

Why this exists (HPD-1097): Xvnc used to run ``-SendCutText=0`` so that no watcher could ever
receive the holder's clipboard, which also made "copy to phone" impossible. Xvnc now sends its
clipboard, and this parser removes every clipboard message that carries content, or announces
that content exists, from the stream of any viewer that does not hold human input right now.
Browser clipboards hold passwords; nothing here logs, stores or returns clipboard bytes.

RFB server messages are not length-prefixed, so the stream is framed message by message
(RFC 6143 §7.6 plus the TigerVNC extensions noVNC negotiates). Framebuffer rectangles are framed
per encoding; to keep that finite, :func:`allowed_encodings` is applied to the client's
SetEncodings by :class:`~tools.bot_desktop.rfb_filter.RfbClientFilter`, so the server can only
use encodings this parser understands (newer noVNC extras such as ExtendedMouseButtons -316 or
H.264 50 are stripped there and only lose a feature). Anything unknown raises: the viewer connection drops and
reconnects, which is the safe failure (a misframed stream could otherwise hide a clipboard
message).

Clipboard rules for a viewer without input:
  * classic ServerCutText (content) — dropped;
  * extended clipboard ``caps`` (caps bit set, whatever other action bits TigerVNC adds) —
    forwarded: capability flags and size limits only, and without it noVNC would
    never use the extended protocol once the viewer takes over (HPD-1141 connects first, takes
    over after);
  * extended ``request`` — forwarded: the server asks the viewer for its clipboard, carries no
    server content, and the viewer's answer is input and stays gated by RfbClientFilter;
  * extended ``notify``, ``provide``, ``peek`` and anything else — dropped.
The holder receives every message unchanged.
"""

from __future__ import annotations

from typing import Callable, Generator, Optional

# Encodings (RFC 6143 §7.7, TigerVNC pseudo-encodings), signed 32-bit.
RAW, COPY_RECT, TIGHT, ZRLE = 0, 1, 7, 16
CURSOR, DESKTOP_SIZE, LAST_RECT = -239, -223, -224
QEMU_EXTENDED_KEY, QEMU_LED = -258, -261
EXTENDED_DESKTOP_SIZE, DESKTOP_NAME, FENCE, CONTINUOUS_UPDATES = -308, -307, -312, -313
EXTENDED_CLIPBOARD = 0xC0A1E5CE - (1 << 32)
QUALITY_LEVELS = range(-32, -22)       # pseudo: no rectangle payload ever
COMPRESS_LEVELS = range(-256, -246)    # pseudo: no rectangle payload ever

_FRAMED = {RAW, COPY_RECT, TIGHT, ZRLE, CURSOR, DESKTOP_SIZE, LAST_RECT, QEMU_EXTENDED_KEY, QEMU_LED,
           EXTENDED_DESKTOP_SIZE, DESKTOP_NAME, FENCE, CONTINUOUS_UPDATES, EXTENDED_CLIPBOARD}


def allowed_encodings(encoding: int) -> bool:
    """True for encodings whose rectangles this parser frames (or that never produce one)."""
    return encoding in _FRAMED or encoding in QUALITY_LEVELS or encoding in COMPRESS_LEVELS


_CAPS, _REQUEST = 1 << 24, 1 << 25
_ACTIONS = 0xFF << 24


class PixelFormat:
    def __init__(self, raw: bytes) -> None:
        if len(raw) != 16:
            raise ValueError("pixel format must be 16 bytes")
        self.bpp, self.depth, _big_endian, self.true_colour = raw[0], raw[1], raw[2], raw[3]
        self.maxes = (int.from_bytes(raw[4:6], "big"), int.from_bytes(raw[6:8], "big"),
                      int.from_bytes(raw[8:10], "big"))
        self.shifts = (raw[10], raw[11], raw[12])
        if self.bpp not in (8, 16, 32):
            raise ValueError("unsupported bits per pixel")

    @property
    def bytes_per_pixel(self) -> int:
        return self.bpp // 8

    @property
    def tight_pixel(self) -> int:
        """Tight's TPIXEL: 3 bytes for an 8-8-8 true-colour 32 bpp format, else a whole pixel."""
        is888 = (self.true_colour and self.bpp == 32 and self.depth == 24 and self.maxes == (255, 255, 255)
                 and all(shift % 8 == 0 for shift in self.shifts))
        return 3 if is888 else self.bytes_per_pixel


class RfbServerFilter:
    """Feed server bytes with :meth:`feed`; get back the bytes this viewer may receive.

    ``allow_clipboard`` is consulted for every clipboard message, at the moment its header is read.
    :meth:`set_pixel_format` follows the client's SetPixelFormat (wired by the bridge).
    """

    def __init__(self, allow_clipboard: Callable[[], bool]) -> None:
        self._allow = allow_clipboard
        self._buf = bytearray()
        self._out = bytearray()
        self._format: Optional[PixelFormat] = None
        self._pending_format: Optional[PixelFormat] = None
        self._steps = self._stream()
        self._need = next(self._steps)
        self._failed = False

    def set_pixel_format(self, raw: bytes) -> None:
        # Takes effect from the next FramebufferUpdate on; Xvnc applies it in order too.
        self._pending_format = PixelFormat(raw)

    def feed(self, chunk: bytes) -> bytes:
        if self._failed:
            raise ValueError("RFB server stream already failed")
        self._buf += chunk
        try:
            while len(self._buf) >= self._need:
                self._need = self._steps.send(None)
        except Exception:
            self._failed = True
            self._buf.clear()
            self._out.clear()
            raise
        out = bytes(self._out)
        self._out.clear()
        return out

    # ---- primitive steps; each ``yield n`` waits until n bytes are buffered ----

    def _take(self, n: int, forward: bool = True) -> Generator[int, None, bytes]:
        while len(self._buf) < n:
            yield n
        data = bytes(self._buf[:n])
        del self._buf[:n]
        if forward:
            self._out += data
        return data

    def _skip(self, n: int, forward: bool) -> Generator[int, None, None]:
        """Pass (or drop) n bytes without holding them, across as many chunks as they span."""
        while n:
            while not self._buf:
                yield 1
            k = min(n, len(self._buf))
            if forward:
                self._out += self._buf[:k]
            del self._buf[:k]
            n -= k

    def _u8(self, forward: bool = True):
        return (yield from self._take(1, forward))[0]

    def _u16(self, forward: bool = True):
        return int.from_bytes((yield from self._take(2, forward)), "big")

    def _u32(self, forward: bool = True):
        return int.from_bytes((yield from self._take(4, forward)), "big")

    def _s32(self, forward: bool = True):
        return int.from_bytes((yield from self._take(4, forward)), "big", signed=True)

    # ---- the stream ----

    def _stream(self):
        if (yield from self._take(12)) != b"RFB 003.008\n":
            raise ValueError("unsupported RFB version")
        count = yield from self._u8()
        if count == 0:
            yield from self._skip((yield from self._u32()), True)
            raise ValueError("server refused the connection")
        yield from self._take(count)
        if (yield from self._u32()) != 0:
            yield from self._skip((yield from self._u32()), True)
            raise ValueError("security handshake failed")
        init = yield from self._take(24)
        self._format = PixelFormat(init[4:20])
        yield from self._skip(int.from_bytes(init[20:24], "big"), True)
        while True:
            kind = yield from self._u8(forward=False)
            if kind == 3:
                yield from self._cut_text()
                continue
            self._out.append(kind)
            if kind == 0:
                yield from self._framebuffer_update()
            elif kind == 1:   # SetColourMapEntries
                yield from self._take(3)
                yield from self._skip(6 * (yield from self._u16()), True)
            elif kind == 2:   # Bell
                pass
            elif kind == 150:  # EndOfContinuousUpdates
                pass
            elif kind == 248:  # ServerFence
                yield from self._take(7)
                yield from self._skip((yield from self._u8()), True)
            else:
                raise ValueError(f"unknown RFB server message type {kind}")

    def _cut_text(self):
        header = yield from self._take(7, forward=False)  # pad3 + length
        length = int.from_bytes(header[3:7], "big", signed=True)
        if length >= 0:
            forward = bool(self._allow())
            if forward:
                self._out += b"\x03" + header
            yield from self._skip(length, forward)
            return
        size = -length
        if size < 4:
            raise ValueError("extended clipboard message too short")
        flags = yield from self._take(4, forward=False)
        actions = int.from_bytes(flags, "big") & _ACTIONS
        # A set caps bit makes the whole message a caps message (TigerVNC sends caps with every action
        # it supports, 0x1F000001; noVNC reads isCaps first and nothing else of it).
        forward = bool(self._allow()) or bool(actions & _CAPS) or actions == _REQUEST
        if forward:
            self._out += b"\x03" + header + flags
        yield from self._skip(size - 4, forward)

    def _framebuffer_update(self):
        if self._pending_format is not None:
            self._format, self._pending_format = self._pending_format, None
        yield from self._take(1)
        rects = yield from self._u16()
        index = 0
        while rects == 0xFFFF or index < rects:
            index += 1
            _x, _y, w, h = [int.from_bytes(part, "big") for part in _split((yield from self._take(8)), 2)]
            encoding = yield from self._s32()
            if encoding == LAST_RECT:
                return
            yield from self._rect(encoding, w, h)

    def _rect(self, encoding: int, w: int, h: int):
        fmt = self._format
        if encoding == RAW:
            yield from self._skip(w * h * fmt.bytes_per_pixel, True)
        elif encoding == COPY_RECT:
            yield from self._take(4)
        elif encoding == ZRLE:
            yield from self._skip((yield from self._u32()), True)
        elif encoding == TIGHT:
            yield from self._tight(w, h)
        elif encoding == CURSOR:
            yield from self._skip(w * h * fmt.bytes_per_pixel + ((w + 7) // 8) * h, True)
        elif encoding in (DESKTOP_SIZE, QEMU_EXTENDED_KEY):
            pass
        elif encoding == QEMU_LED:
            yield from self._take(1)
        elif encoding == EXTENDED_DESKTOP_SIZE:
            screens = (yield from self._take(4))[0]
            yield from self._skip(16 * screens, True)
        elif encoding == DESKTOP_NAME:
            yield from self._skip((yield from self._u32()), True)
        else:
            raise ValueError(f"unframed RFB encoding {encoding}")

    def _compact_length(self):
        length = 0
        for shift in (0, 7, 14):
            byte = yield from self._u8()
            length |= (byte & (0x7F if shift < 14 else 0xFF)) << shift
            if shift == 14 or not byte & 0x80:
                break
        return length

    def _tight(self, w: int, h: int):
        tpix = self._format.tight_pixel
        control = (yield from self._u8()) >> 4
        if control == 0x08:          # fill
            yield from self._take(tpix)
            return
        if control == 0x09:          # JPEG
            yield from self._skip((yield from self._compact_length()), True)
            return
        if control > 0x09:           # PNG exists only in TightPNG, which is never negotiated
            raise ValueError("unsupported tight compression")
        row = w * tpix
        if control & 0x04:
            selected = yield from self._u8()
            if selected == 1:        # palette
                colours = (yield from self._u8()) + 1
                yield from self._take(colours * tpix)
                row = (w + 7) // 8 if colours <= 2 else w
            elif selected not in (0, 2):  # 0 copy, 2 gradient: both whole TPIXELs per pixel
                raise ValueError("unsupported tight filter")
        size = row * h
        if size < 12:
            yield from self._take(size)
        else:
            yield from self._skip((yield from self._compact_length()), True)


def _split(data: bytes, width: int):
    return [data[i:i + width] for i in range(0, len(data), width)]
