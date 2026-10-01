"""Actual cryptographic binding, replay fence and native RFB input gate."""
import os
import struct
import time

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from tools.bot_desktop import lease
from tools.bot_desktop.rfb_filter import RfbClientFilter
from tools.isolated_browser import BrowserBoundaryError, IsolatedBrowserBoundary
from tools.isolated_browser_viewer import NativeRfbViewer, RfbCipher


def public(key):
    return key.public_key().public_bytes(serialization.Encoding.X962,
                                        serialization.PublicFormat.UncompressedPoint)


def pair(client_session="run", client_viewer="viewer"):
    server_key, client_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    server = RfbCipher(server_key, public(client_key), session_id="run", viewer_id="viewer", server=True)
    client = RfbCipher(client_key, public(server_key), session_id=client_session, viewer_id=client_viewer, server=False)
    return server, client


def test_encrypted_native_rfb_has_distinct_direction_nonces_and_rejects_replay():
    server, client = pair()
    packet = client.encrypt(b"synthetic input")
    assert b"synthetic input" not in packet
    assert server.decrypt(packet) == b"synthetic input"
    with pytest.raises(BrowserBoundaryError, match="frame_replay"):
        server.decrypt(packet)
    response = server.encrypt(b"framebuffer")
    assert client.decrypt(response) == b"framebuffer"
    with pytest.raises(BrowserBoundaryError, match="frame_replay"):
        server.decrypt(response)


@pytest.mark.parametrize("session,viewer", [("foreign-run", "viewer"), ("run", "foreign-viewer")])
def test_cipher_binds_exact_session_and_viewer(session, viewer):
    server, client = pair(session, viewer)
    with pytest.raises(InvalidTag):
        server.decrypt(client.encrypt(b"test"))


def test_tamper_does_not_advance_receive_counter():
    server, client = pair()
    original = client.encrypt(b"test")
    tampered = original[:-1] + bytes([original[-1] ^ 1])
    with pytest.raises(InvalidTag):
        server.decrypt(tampered)
    assert server.decrypt(original) == b"test"


def test_out_of_order_and_oversize_frames_fail_closed():
    server, client = pair()
    first = client.encrypt(b"1")
    second = client.encrypt(b"2")
    with pytest.raises(BrowserBoundaryError, match="frame_replay"):
        server.decrypt(second)
    assert server.decrypt(first) == b"1"
    assert server.decrypt(second) == b"2"
    with pytest.raises(BrowserBoundaryError, match="invalid_frame"):
        client.encrypt(b"x" * 65537)


@pytest.fixture
def viewer():
    boundary = IsolatedBrowserBoundary(lease, lambda *args: {}, lambda: True,
        agent_uid=os.geteuid()+1, management_uid=os.geteuid(), session_id="run", expires_at=time.time()+300)
    return NativeRfbViewer(boundary, "/unused")


def test_ticket_single_use_short_ttl_and_native_lease(viewer):
    minted = viewer.mint("viewer")
    assert time.time() < minted["expires_at"] <= time.time()+60
    assert viewer.consume(minted["ticket"]).viewer_id == "viewer"
    with pytest.raises(BrowserBoundaryError, match="invalid_ticket"):
        viewer.consume(minted["ticket"])
    viewer.boundary.dispatch(viewer.boundary.management_uid, {"session_id": "run", "epoch": 0,
        "method": "display.lease.acquire", "viewer_id": "viewer"})
    assert viewer.may_view("viewer") and not viewer.may_view("observer")
    with pytest.raises(BrowserBoundaryError, match="viewer_mismatch"):
        viewer.mint("observer")


def test_ticket_expiry_and_cancel(viewer):
    minted = viewer.mint("viewer")
    viewer.tickets[minted["ticket"]].expires_at = 0
    with pytest.raises(BrowserBoundaryError, match="invalid_ticket"):
        viewer.consume(minted["ticket"])
    viewer.boundary.revoked = True
    assert not viewer.may_view("viewer")
    with pytest.raises(BrowserBoundaryError, match="session_expired"):
        viewer.mint("viewer")


def test_native_filter_gates_fragmented_keyboard_clipboard_and_extended_keys(viewer):
    gate = RfbClientFilter(lambda: viewer.boundary.viewer_may_input("viewer"))
    assert gate.feed(b"RFB 003.008\n\x01\x00") == b"RFB 003.008\n\x01\x01"
    key = struct.pack(">BBHI", 4, 1, 0, ord("a"))
    clipboard = b"\x06\x00\x00\x00" + struct.pack(">i", 4) + b"test"
    extended = b"\xff\x00\x00\x01" + b"\x00" * 8
    assert gate.feed(key[:3]) == b""
    assert gate.feed(key[3:] + clipboard + extended) == b""
    viewer.boundary.dispatch(viewer.boundary.management_uid, {"session_id": "run", "epoch": 0,
        "method": "display.lease.acquire", "viewer_id": "viewer"})
    assert gate.feed(key + clipboard + extended) == key + clipboard + extended
    viewer.boundary.ready_viewer = None
    assert gate.feed(key) == b""
