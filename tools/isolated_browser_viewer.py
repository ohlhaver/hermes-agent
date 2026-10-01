"""Encrypted transport for the native Bot Desktop RFB stream.

Tickets are minted only on the private management socket, consumed once, and
never appear in URLs. A Control Plane tunnel relays ciphertext. Input still
uses the native byte-level RFB parser and the broker's drained human lease.
"""
from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import secrets
import threading
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from tools.bot_desktop.rfb_filter import RfbClientFilter
from tools.isolated_browser import BrowserBoundaryError

MAX_FRAME = 65536


class RfbCipher:
    def __init__(self, private_key, peer_public: bytes, *, session_id: str, viewer_id: str, server: bool):
        peer = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), peer_public)
        self.context = b"hpd321-rfb-v1\0" + session_id.encode() + b"\0" + viewer_id.encode()
        key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=self.context).derive(
            private_key.exchange(ec.ECDH(), peer))
        self.aes = AESGCM(key)
        self.out_prefix = b"HHS1" if server else b"HHC1"
        self.in_prefix = b"HHC1" if server else b"HHS1"
        self.send_counter = self.receive_counter = 0

    def encrypt(self, plaintext: bytes) -> bytes:
        if len(plaintext) > MAX_FRAME:
            raise BrowserBoundaryError("invalid_frame")
        nonce = self.out_prefix + self.send_counter.to_bytes(8, "big")
        self.send_counter += 1
        return nonce + self.aes.encrypt(nonce, plaintext, self.context)

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not isinstance(ciphertext, bytes) or not 28 <= len(ciphertext) <= MAX_FRAME + 28:
            raise BrowserBoundaryError("invalid_frame")
        expected = self.in_prefix + self.receive_counter.to_bytes(8, "big")
        if ciphertext[:12] != expected:
            raise BrowserBoundaryError("frame_replay")
        plaintext = self.aes.decrypt(expected, ciphertext[12:], self.context)
        self.receive_counter += 1
        return plaintext


@dataclass
class ViewerTicket:
    private_key: object
    viewer_id: str
    expires_at: float


class NativeRfbViewer:
    def __init__(self, boundary, socket_path):
        self.boundary = boundary
        self.socket_path = socket_path
        self.tickets = {}
        self.lock = threading.Lock()
        self.connected = None

    def mint(self, viewer_id: str):
        if not isinstance(viewer_id, str) or not viewer_id or len(viewer_id) > 256 or "\0" in viewer_id:
            raise BrowserBoundaryError("invalid_viewer")
        with self.boundary.condition:
            if self.boundary.revoked or time.time() >= self.boundary.expires_at:
                raise BrowserBoundaryError("session_expired")
            current = self.boundary.lease.get()
            if current.holder == self.boundary.lease.HUMAN and self.boundary.viewer_id != viewer_id:
                raise BrowserBoundaryError("viewer_mismatch")
            private = ec.generate_private_key(ec.SECP256R1())
            token = secrets.token_urlsafe(32)
            expires = min(time.time() + 60, self.boundary.expires_at)
            with self.lock:
                self.tickets = {k: v for k, v in self.tickets.items() if v.expires_at > time.time()}
                if len(self.tickets) >= 4:
                    raise BrowserBoundaryError("viewer_ticket_limit")
                self.tickets[token] = ViewerTicket(private, viewer_id, expires)
            public = private.public_key().public_bytes(serialization.Encoding.X962,
                                                       serialization.PublicFormat.UncompressedPoint)
            return {"ticket": token, "server_public_key": base64.b64encode(public).decode(),
                    "expires_at": expires, "session_id": self.boundary.session_id,
                    "viewer_id": viewer_id, "protocol": "hpd321-rfb-v1"}

    def consume(self, token):
        if not isinstance(token, str) or len(token) > 256:
            raise BrowserBoundaryError("invalid_ticket")
        with self.lock:
            ticket = self.tickets.pop(token, None)
        if ticket is None or ticket.expires_at <= time.time():
            raise BrowserBoundaryError("invalid_ticket")
        return ticket

    def may_view(self, viewer_id):
        with self.boundary.condition:
            if self.boundary.revoked or time.time() >= self.boundary.expires_at:
                return False
            return (self.boundary.lease.get().holder == self.boundary.lease.AGENT
                    or self.boundary.viewer_id == viewer_id)

    async def handle(self, ws):
        reader = writer = None
        tasks = []
        viewer_id = None
        try:
            hello = await asyncio.wait_for(ws.recv(), 10)
            if not isinstance(hello, str) or len(hello) > 2048:
                raise BrowserBoundaryError("invalid_handshake")
            request = json.loads(hello)
            ticket = self.consume(request.get("ticket"))
            viewer_id = ticket.viewer_id
            cipher = RfbCipher(ticket.private_key, base64.b64decode(request["client_public_key"], validate=True),
                               session_id=self.boundary.session_id, viewer_id=viewer_id, server=True)
            if self.connected is not None or not self.may_view(viewer_id):
                raise BrowserBoundaryError("viewer_unavailable")
            self.connected = ws
            reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
            await ws.send('{"ready":true}')
            gate = RfbClientFilter(lambda: self.boundary.viewer_may_input(viewer_id))

            async def to_browser():
                async for packet in ws:
                    if not self.may_view(viewer_id):
                        return
                    with self.boundary.condition:
                        allowed = gate.feed(cipher.decrypt(packet))
                        if allowed:
                            writer.write(allowed)
                    if allowed:
                        await writer.drain()

            async def to_viewer():
                while self.may_view(viewer_id):
                    data = await reader.read(MAX_FRAME)
                    if not data:
                        return
                    if not self.may_view(viewer_id):
                        return
                    await ws.send(cipher.encrypt(data))

            async def expires():
                while self.may_view(viewer_id):
                    await asyncio.sleep(.5)

            tasks = [asyncio.create_task(to_browser()), asyncio.create_task(to_viewer()),
                     asyncio.create_task(expires())]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception:
            # Never include a token, handshake, input/frame bytes or URL.
            pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            if self.connected is ws:
                self.connected = None
                with self.boundary.condition:
                    if self.boundary.ready_viewer == viewer_id:
                        self.boundary.ready_viewer = None
                        # Disconnect is NEVER an implicit release to the agent.
            await ws.close(code=1000, reason="Viewer disconnected")
