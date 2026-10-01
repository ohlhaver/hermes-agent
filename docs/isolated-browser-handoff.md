# Isolated native browser handoff (HPD-321)

This source checkpoint targets the accepted fork commit
`d0625482c31b62e75da73865ed19558d9707c9df`. It does not update the default fork
or import the newer upstream browser refactor. Native Bot Desktop components
come from `f97608f178d1ffeca59860195ab7da295f7c8e5f` with only accepted-base
helper adapters in `browser.py`, `lease.py` and `install.py`. Original authors
are credited in the backport commit.

## Process boundary

Run `python -m tools.isolated_browser_service <manifest>` as a browser UID
different from the runtime agent. The service owns the native Xvnc/Xfce screen,
the dock's persistent Chromium profile and CDP endpoint. Existing Hermes
browser tools dispatch to `browser.isolated_socket` and
`browser.isolated_session_id` in `config.yaml`; they do not start another local,
cloud or Camofox browser when this binding is configured. The broker executes
the existing native `agent-browser` command implementation against the dock's
Chromium.

The required deployment has **separate mount, PID and network namespaces**.
UID separation alone cannot isolate a loopback CDP endpoint. The agent gets
only the agent RPC directory through a read-only mount, never the profile,
display/Xauthority/RFB sockets, Docker socket or management socket. Provision
the socket directory with broker ownership and the agent group, mode 0750;
the broker belongs to that supplementary group and creates its socket 0660.
The management socket is in an unshared directory. `SO_PEERCRED` and the
socket role independently restrict callers. The production runtime UID is
10001, so its intended separate browser UID is 10002. Probe UID numbers are
not production pins.

The owner-written 0600 manifest specifies `session_id`, `expires_at` (at most
30 minutes), `agent_uid`, `management_uid`, `agent_group`, `agent_socket`,
`management_socket` and `viewer_port`. Configure the browser's private Hermes
home for local Chromium, headed mode and no recording. Set its process home,
private TMPDIR and XDG runtime directory correctly for its real passwd entry.
Bake TigerVNC/Xfce/Chromium and exact agent-browser 0.26.0 dependencies into
the product's isolated image. The probe image is not a production artifact.
Restrict network egress independently, including private addresses and DNS
rebinding; URL preflight checks alone cannot enforce that boundary.

Account, workspace, authorized run and authentication-session checks belong to
the trusted product/Node-Agent binding. Provision one live, exclusively owned
browser session; do not mount its RPC endpoint into a different run's worker.
The agent-facing session ID is a non-secret binding, not a management
capability. Product integration must revoke the process/session on logout,
account switch, run cancellation and expiry. These product checks are not
claimed by the native prototype.

## Handoff and viewer

The native file lease remains authority. Human acquire fences new agent
commands immediately, discards late results and drains already admitted work
before allowing RFB input. Explicit handback fences human input and clears
input/textarea values through isolated CDP worlds and clears the native remote
clipboard before releasing the lease. Otherwise a site's own Paste button
could reveal residual human clipboard data after return.
Failure retains human ownership and allows the same viewer to correct the
page. Console/error buffers, arbitrary JS, HTML/form-value getters, recordings,
filesystem export and DevTools shortcuts are unavailable through agent RPC.
Simple existing navigation, snapshots, references, forms, scrolling and key
actions remain. Screenshot/image-tool transport is not implemented in this
checkpoint; that omission must not be presented as full browser acceptance.

Only the private management role mints single-use, 60-second viewer tickets.
They are sent in the first WebSocket message, never the URL. Each ticket has
an ephemeral P-256 key. Client/server derive AES-256-GCM through ECDH and
HKDF-SHA256 with the context
`hpd321-rfb-v1\0<session_id>\0<viewer_id>`. The 12-byte nonce is `HHC1` (client)
or `HHS1` (server) followed by a big-endian 64-bit counter. The packet is nonce
then authenticated ciphertext; exact increasing counters reject replay,
reordering and cross-direction substitution. The context is also authenticated
additional data. A Control Plane tunnel forwards ciphertext, without
terminating the RFB input/frame encryption.

The decrypted stream goes through the upstream native `RfbClientFilter`;
the actual lease gates keyboard, pointer, clipboard and resize messages.
Only one viewer connection is served. A different viewer cannot observe a
human-held screen. Disconnect clears input readiness and **never** releases
control to Hermes; reacquire is explicit. Expiry/cancel revoke the endpoint
and stop the private desktop. The product must expose target domain/current
browser location, takeover, return and cancellation clearly. The browser's
address bar remains inside the encrypted framebuffer.

## Verified evidence and remaining delivery

`scripts/probe_isolated_browser.py` operates only explicitly named owned
HPD321 test containers. It captures all private subprocess/RPC payloads and
prints booleans. The recorded Linux probe used an ARM64 desktop dependency
image `sha256:06df1772bb64ebaf87c0cd884f4006328ef7a838f43d761319456ddb45266f33`,
the accepted native source plus this overlay, exact agent-browser 0.26.0, two
separate containers and a read-only RPC volume. The desktop dependency image
has newer provider entrypoint metadata than the accepted fork; model calls
were not made and this image is not a release candidate.

The real Chromium/RFB probe passed navigation and snapshot, native human
lease, real encrypted keyboard input into a synthetic password/2FA form,
same-profile login/cookie continuity, agent observation rejection during
takeover, explicit safe handback, continued navigation state, absence of the
synthetic input in agent output and cancellation fencing. Focus tests cover
lease races, failed return, malformed/foreign identity, restricted commands,
no backend fallback, crypto context, tamper/replay and native RFB framing.

This is **native source/prototype evidence**. Product/API session binding,
the actual web/mobile viewer and tunnel, shipped image/Compose/policy,
independent review, CI, TestFlight/test URL, real model-driven browsing,
real iPhone AutoFill/passkeys and anti-bot behavior on representative websites
remain open. RFB alone does not establish iPhone password-manager or passkey
support. HPD-321 remains active through usable test delivery.
