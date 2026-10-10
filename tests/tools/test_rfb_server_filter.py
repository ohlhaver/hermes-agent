"""HPD-1097: the clipboard Xvnc now sends reaches only the pane that holds human input."""
import os
import struct
import time

import pytest

from tools.bot_desktop import lease
from tools.bot_desktop.rfb_filter import RfbClientFilter
from tools.bot_desktop.rfb_server_filter import (
    COPY_RECT, CURSOR, DESKTOP_NAME, DESKTOP_SIZE, EXTENDED_CLIPBOARD, EXTENDED_DESKTOP_SIZE, LAST_RECT,
    QEMU_EXTENDED_KEY, QEMU_LED, RAW, TIGHT, ZRLE, RfbServerFilter, allowed_encodings)
from tools.isolated_browser import IsolatedBrowserBoundary
from tools.isolated_browser_viewer import NativeRfbViewer

SECRET = b"synthetic-password-1097"
FORMAT_888 = struct.pack(">BBBBHHHBBBxxx", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
FORMAT_565 = struct.pack(">BBBBHHHBBBxxx", 16, 16, 0, 1, 31, 63, 31, 11, 5, 0)


def handshake(fmt=FORMAT_888, name=b"hermes:browser"):
    return (b"RFB 003.008\n" + b"\x01\x01" + b"\x00\x00\x00\x00"
            + struct.pack(">HH", 1440, 900) + fmt + struct.pack(">I", len(name)) + name)


def rect(x, y, w, h, encoding, payload=b""):
    return struct.pack(">HHHHi", x, y, w, h, encoding) + payload


def update(*rects, last_rect=False):
    count = 0xFFFF if last_rect else len(rects)
    body = b"".join(rects) + (rect(0, 0, 0, 0, LAST_RECT) if last_rect else b"")
    return b"\x00\x00" + struct.pack(">H", count) + body


def compact(n):
    out = bytearray([n & 0x7F])
    if n > 0x7F:
        out[0] |= 0x80; out.append((n >> 7) & 0x7F)
        if n > 0x3FFF:
            out[1] |= 0x80; out.append((n >> 14) & 0xFF)
    return bytes(out)


def cut_text(text):
    return b"\x03\x00\x00\x00" + struct.pack(">i", len(text)) + text


def extended(action_flags, payload=b""):
    body = struct.pack(">I", action_flags | 1) + payload  # format bit 0: text
    return b"\x03\x00\x00\x00" + struct.pack(">i", -len(body)) + body


CAPS, REQUEST, PEEK, NOTIFY, PROVIDE = 1 << 24, 1 << 25, 1 << 26, 1 << 27, 1 << 28
FRAMES = update(
    rect(0, 0, 4, 2, RAW, b"\x11" * 4 * 2 * 4),
    rect(4, 0, 4, 2, COPY_RECT, b"\x00\x01\x00\x02"),
    rect(0, 2, 8, 8, ZRLE, struct.pack(">I", 300) + b"\x7a" * 300),
    rect(0, 0, 16, 16, TIGHT, b"\x80" + b"\x01\x02\x03"),                              # fill
    rect(0, 0, 16, 16, TIGHT, b"\x90" + compact(200) + b"\xff" * 200),                  # JPEG
    rect(0, 0, 2, 1, TIGHT, b"\x00" + b"\x05" * 6),                                     # basic copy, < 12 raw
    rect(0, 0, 64, 64, TIGHT, b"\x40\x01\x01" + b"\x00" * 6 + compact(70) + b"\x09" * 70),  # 2-colour palette
    rect(0, 0, 64, 64, TIGHT, b"\x40\x01\x0f" + b"\x00" * 48 + compact(20_000) + b"\x08" * 20_000),  # 16 colours
    rect(0, 0, 64, 64, TIGHT, b"\x40\x02" + compact(5000) + b"\x07" * 5000),             # gradient
    rect(0, 0, 3, 2, TIGHT, b"\x40\x01\x01" + b"\x00" * 6 + b"\x01\x02"),                # palette, 2 bytes raw
    rect(1, 1, 9, 3, CURSOR, b"\x22" * 9 * 3 * 4 + b"\x33" * 2 * 3),
    rect(0, 0, 1440, 900, DESKTOP_SIZE),
    rect(0, 0, 0, 0, QEMU_EXTENDED_KEY),
    rect(0, 0, 0, 0, QEMU_LED, b"\x02"),
    rect(0, 0, 1440, 900, EXTENDED_DESKTOP_SIZE, b"\x01\x00\x00\x00" + b"\x00" * 16),
    rect(0, 0, 0, 0, DESKTOP_NAME, struct.pack(">I", 5) + b"hello"),
    last_rect=True)
OTHER = (b"\x02"                                                    # Bell
         + b"\x01\x00\x00\x00\x00\x02" + b"\x00" * 12               # SetColourMapEntries
         + b"\x96"                                                  # EndOfContinuousUpdates
         + b"\xf8\x00\x00\x00" + struct.pack(">I", 1) + b"\x03abc")  # ServerFence


def stream_with_clipboard():
    content = [cut_text(SECRET), extended(NOTIFY), extended(PROVIDE, b"\x00\x00\x00\x20" + SECRET),
               extended(PEEK)]
    shareable = [extended(CAPS, struct.pack(">I", 1 << 20)), extended(REQUEST)]
    stream = handshake() + FRAMES + content[0] + OTHER + shareable[0] + FRAMES + b"".join(content[1:]) + shareable[1] + FRAMES
    without = handshake() + FRAMES + OTHER + shareable[0] + FRAMES + shareable[1] + FRAMES
    return stream, without


def run(stream, allow, chunk=None):
    gate = RfbServerFilter(lambda: allow)
    if chunk is None:
        return gate.feed(stream)
    return b"".join(gate.feed(stream[i:i + chunk]) for i in range(0, len(stream), chunk))


@pytest.mark.parametrize("chunk", [None, 1, 7, 4096])
def test_holder_receives_every_byte_unchanged(chunk):
    stream, _ = stream_with_clipboard()
    assert run(stream, True, chunk) == stream


@pytest.mark.parametrize("chunk", [None, 1, 7, 4096])
def test_watcher_never_receives_clipboard_content_but_keeps_frames_and_caps(chunk):
    stream, without = stream_with_clipboard()
    out = run(stream, False, chunk)
    assert SECRET not in out
    assert out == without


def test_decision_follows_the_lease_per_message():
    allowed = {"now": False}
    gate = RfbServerFilter(lambda: allowed["now"])
    assert gate.feed(handshake()) == handshake()
    assert gate.feed(cut_text(SECRET)) == b""
    allowed["now"] = True
    assert gate.feed(cut_text(SECRET)) == cut_text(SECRET)
    allowed["now"] = False
    assert gate.feed(cut_text(SECRET)) == b""


def test_large_clipboard_is_streamed_or_dropped_without_buffering():
    big = b"x" * 3_000_000
    gate = RfbServerFilter(lambda: False)
    gate.feed(handshake())
    out = b"".join(gate.feed(part) for part in (cut_text(big)[i:i + 65536] for i in range(0, len(big) + 8, 65536)))
    assert out == b""
    assert len(gate._buf) == 0


def test_pixel_format_from_the_client_frames_later_updates():
    gate = RfbServerFilter(lambda: False)
    gate.feed(handshake(FORMAT_565))
    gate.set_pixel_format(FORMAT_888)
    raw888 = update(rect(0, 0, 2, 2, RAW, b"\x01" * 16)) + cut_text(SECRET) + b"\x02"
    assert gate.feed(raw888) == update(rect(0, 0, 2, 2, RAW, b"\x01" * 16)) + b"\x02"
    # 16 bpp: Tight's TPIXEL is the whole 2-byte pixel.
    gate = RfbServerFilter(lambda: False)
    gate.feed(handshake(FORMAT_565))
    fill = update(rect(0, 0, 8, 8, TIGHT, b"\x80\x01\x02")) + cut_text(SECRET)
    assert gate.feed(fill) == update(rect(0, 0, 8, 8, TIGHT, b"\x80\x01\x02"))


@pytest.mark.parametrize("bad", [
    update(rect(0, 0, 4, 4, 5)),                        # Hextile is never negotiated
    update(rect(0, 0, 4, 4, -260, b"\xa0")),            # TightPNG
    update(rect(0, 0, 4, 4, TIGHT, b"\xa0")),           # PNG compression
    update(rect(0, 0, 4, 4, TIGHT, b"\x40\x07")),       # unknown tight filter
    b"\xfa\x00\x00\x00",                                # xvp server message
])
def test_anything_unframed_fails_closed(bad):
    gate = RfbServerFilter(lambda: True)
    gate.feed(handshake())
    with pytest.raises(ValueError):
        gate.feed(bad)
    with pytest.raises(ValueError):
        gate.feed(b"\x02")


def test_client_gate_offers_only_encodings_the_server_gate_frames_and_reports_pixel_format():
    seen = []
    gate = RfbClientFilter(lambda: False, allow_encoding=allowed_encodings, on_pixel_format=seen.append)
    gate.feed(b"RFB 003.008\n\x01\x00")
    noVNC = [1, 7, -260, 16, 21, 5, 2, 0, -32 + 6, -256 + 2, -223, -224, -258, -261, -308, -309, -312, -313,
             -307, EXTENDED_CLIPBOARD, 0x574D5664, -239]
    set_encodings = b"\x02\x00" + struct.pack(">H", len(noVNC)) + b"".join(struct.pack(">i", e) for e in noVNC)
    pixel = b"\x00\x00\x00\x00" + FORMAT_888
    out = gate.feed(pixel + set_encodings)
    kept = [e for e in noVNC if e not in (-260, 21, 5, 2, -309, 0x574D5664)]
    assert out == pixel + b"\x02\x00" + struct.pack(">H", len(kept)) + b"".join(struct.pack(">i", e) for e in kept)
    assert seen == [FORMAT_888]
    # Without the options the stream is exactly as before.
    plain = RfbClientFilter(lambda: False)
    plain.feed(b"RFB 003.008\n\x01\x00")
    assert plain.feed(pixel + set_encodings) == pixel + set_encodings


@pytest.fixture
def boundary():
    return IsolatedBrowserBoundary(lease, lambda *args: {}, lambda: True, agent_uid=os.geteuid() + 1,
                                   management_uid=os.geteuid(), session_id="run", expires_at=time.time() + 300)


def acquire(boundary, pane):
    return boundary.dispatch(boundary.management_uid, {"session_id": "run", "epoch": boundary.lease.get().epoch,
                                                       "method": "display.lease.acquire", "viewer_id": pane})


def test_watcher_versus_lease_holder_on_the_real_boundary(boundary):
    stream, without = stream_with_clipboard()
    watcher = lambda: RfbServerFilter(lambda: boundary.viewer_may_input("pane-a"))
    assert watcher().feed(stream) == without, "agent holds the lease: the pane only watches"
    acquire(boundary, "pane-a")
    assert RfbServerFilter(lambda: boundary.viewer_may_input("observer")).feed(stream) == without
    assert watcher().feed(stream) == stream, "the customer's pane holds input"
    boundary.ready_viewer = None
    assert watcher().feed(stream) == without, "input not ready (e.g. in a transition): no clipboard"


def test_replaced_pane_after_hpd1141_gets_no_clipboard(boundary):
    """HPD-1141: a reopened pane replaces the old one under the same native viewer id.

    The old connection stays shut out even when the new pane holds input again."""
    viewer = NativeRfbViewer(boundary, "/unused")
    old_ws, new_ws = object(), object()
    acquire(boundary, "viewer")
    viewer.connected = old_ws
    old = RfbServerFilter(lambda: viewer.may_receive_clipboard(old_ws, "viewer"))
    new = RfbServerFilter(lambda: viewer.may_receive_clipboard(new_ws, "viewer"))
    old.feed(handshake()); new.feed(handshake())
    assert old.feed(cut_text(SECRET)) == cut_text(SECRET)
    # The old pane's stream closes (readiness cleared), the new one connects and takes over.
    boundary.ready_viewer = None
    viewer.connected = new_ws
    assert old.feed(cut_text(SECRET)) == b""
    assert new.feed(cut_text(SECRET)) == b"", "connected, but not yet taken over"
    acquire(boundary, "viewer")
    assert boundary.viewer_may_input("viewer")
    assert old.feed(cut_text(SECRET)) == b"", "same viewer id, replaced connection"
    assert new.feed(cut_text(SECRET)) == cut_text(SECRET)
    # Handing back to Hermes ends it for the current pane as well.
    boundary.dispatch(boundary.management_uid, {"session_id": "run", "epoch": boundary.lease.get().epoch,
                                                "method": "display.lease.release", "viewer_id": "viewer"})
    assert new.feed(cut_text(SECRET)) == b""
