# Sidecar Protocol (U1)

The sidecar is a long-running local process that holds one `AffordanceEngine` and serves generated SFX
to a game engine over a socket. A game client sends **context changes** (equipped item, surfaces,
region) and **events** (footstep, shoot, ...); for each event the server returns the chosen PCM clip plus
metadata. Defaults to the **mock** generator so the whole stack runs with no GPU.

- **Transport:** loopback TCP, `127.0.0.1`, default port **8765** (tests use port `0` for an ephemeral port).
- **Endianness:** everything little-endian (x86, Unity, C# `BinaryReader` defaults).
- **Server:** `predictive_sfx.sidecar.serve(...)`. Reference Python client: `predictive_sfx.sidecar.SfxClient`.
- **Run it:** `python scripts/run_sidecar.py [--generator mock|sa3] [--port 8765] [--pcm f32le|i16le]`.

## Frame layout

Every message is one length-prefixed frame: a fixed **16-byte header**, then a UTF-8 JSON body, then an
optional raw PCM tail (only on `audio` replies).

| Offset | Size | Field | Notes |
|-------:|-----:|-------|-------|
| 0  | 4 | magic     | ASCII `PSFX` |
| 4  | 1 | version   | `1` |
| 5  | 1 | msg_type  | see table below |
| 6  | 1 | flags     | bit0 = `has_pcm` (a PCM tail follows the JSON) |
| 7  | 1 | reserved  | `0` |
| 8  | 4 | json_len  | uint32, byte length of the JSON body |
| 12 | 4 | pcm_len   | uint32, byte length of the PCM tail (`0` unless `has_pcm`) |

`struct` format: `<4sBBBBII`. Total frame = `16 + json_len + pcm_len`.

**Read rule (both sides):** read exactly N bytes in a loop (a single `recv`/`Read` can return short).
A `recv` returning 0 mid-frame means the peer closed — drop the connection. The reader validates
`json_len <= 65536` and `pcm_len <= 4194304` **from the header before allocating**, so a forged length
cannot trigger a huge allocation. `pcm_len` is authoritative for the tail length.

## Message types

| Opcode | Name | Dir | Payload |
|-------:|------|-----|---------|
| 1  | hello        | C→S | – |
| 2  | welcome      | S→C | – |
| 3  | context      | C→S | – |
| 4  | context_ack  | S→C | – |
| 5  | event        | C→S | – |
| 6  | audio        | S→C | **PCM** |
| 7  | prebake      | C→S | – |
| 8  | prebake_ack  | S→C | – |
| 9  | stats        | C→S | – |
| 10 | stats_reply  | S→C | – |
| 11 | ping         | C→S | – |
| 12 | pong         | S→C | – |
| 255| error        | S→C | – |

## Message bodies (JSON)

```jsonc
// hello (first frame on a connection; wait for welcome before anything else)
{ "type":"hello", "proto":1, "client":"unity", "pcm":"f32le" }      // pcm: "f32le" (default) | "i16le"
// welcome
{ "type":"welcome", "proto":1, "server_version":"0.1.0", "generator":"mock",
  "sample_rate":48000, "channels":1, "pcm":"f32le", "max_pcm_bytes":4194304 }

// context  ->  context_ack
{ "type":"context", "time_s":0.0, "equipped":"gun", "surfaces":["grass"],
  "region":"field", "gpu_pressure":1.0 }                            // equipped may be null
{ "type":"context_ack", "spec_gen":21, "evicted":0, "cached_keys":3 }

// event  ->  audio (+ PCM tail of byte_len bytes)
{ "type":"event", "event_type":"shoot", "material":"metal", "intensity":"normal", "context":"default" }
{ "type":"audio", "served":"cached", "variant_id":8, "format":"f32le", "sample_rate":48000,
  "channels":1, "n_samples":8640, "byte_len":34560, "duration_s":0.18 }

// prebake (level-load warm-up; up to 256 contexts, each warmed under its own lock so events interleave)
{ "type":"prebake", "contexts":[ { "time_s":0.0, "equipped":"gun", "surfaces":["grass"], "region":"field" } ] }
{ "type":"prebake_ack", "spec_gen":21, "peak_bytes":2654208, "cached_keys":3 }

// stats  ->  stats_reply
{ "type":"stats" }
{ "type":"stats_reply", "stats":{ "spec_gen":21,"demand_gen":2,"hits":24,"fallback":2,
  "evicted":0,"gpu_ms":9039.0,"peak_bytes":2654208 }, "wasted":1, "cached_keys":3 }

// ping  ->  pong   (answered without the engine lock; works even during a generate)
{ "type":"ping", "t":123.5 }
{ "type":"pong", "t":123.5, "server_t":1750000000.0 }

// error
{ "type":"error", "code":"oversized", "message":"...", "fatal":true }
```

`served` is `"cached"` (sub-ms, served from the speculative cache) or `"fallback"` (demand-generated now;
~400 ms with `sa3`). Treat event latency as bimodal — prebake/context-warming makes fallbacks rare.

**Error codes:** `bad_frame`, `bad_version`, `oversized` (all `fatal:true` — connection closes after the
error); `bad_json`, `bad_fields`, `unknown_type`, `internal` (all `fatal:false` — the connection stays
open). A client should read the `fatal` field and only reconnect on a fatal error — a non-fatal error
leaves the socket at a clean frame boundary, ready for the next request.

## Connection lifecycle

Connections are persistent and **idle-tolerant**: the server blocks indefinitely waiting for the next
request, so a client may sit silent (menu, cutscene, slow level) without being dropped. `recv_timeout`
(default 30 s) only bounds a *mid-frame* stall — a peer that sends a partial frame and then hangs is reaped.
`SO_KEEPALIVE` is enabled so the OS eventually detects a dead (half-open) peer. Clients do not need an
application-level keepalive, but a periodic `ping` is a cheap liveness/RTT probe (answered without the
engine lock). The warmed cache lives on the long-running server, so it persists across client reconnects.

## PCM contract

- **f32le (default):** raw `float32` little-endian, mono, in `[-1, 1]`. `n_samples = byte_len / 4`. The
  engine already emits `float[]` in `[-1,1]`, so Unity feeds it straight into `AudioClip.SetData` — zero
  conversion.
- **i16le (optional, negotiated in `hello`):** `int16` little-endian, mono. Encode `round(x * 32767)`,
  decode `x / 32767` (edges bit-exact). `n_samples = byte_len / 2`. Halves wire bytes; for bandwidth-
  constrained / remote futures.

Always trust the reply JSON's `format` / `n_samples` / `sample_rate`; never guess. `byte_len == pcm_len`
and equals `n_samples * itemsize` as an independent cross-check.

## Versioning & security

- `proto` is a single int. Mismatch ⇒ `bad_version` + close. New JSON fields may be added without a bump
  (clients must ignore unknown fields); a wire-layout change bumps the version.
- v1 has **no auth/encryption**. The only gate is the `127.0.0.1`-only bind + the `PSFX` magic — any local
  process can connect. **Never bind `0.0.0.0`.** A `token` hello field is reserved for a future
  networked/hostile-local threat model.

## Clients

- **Python (reference / mock):** `predictive_sfx.sidecar.SfxClient` — `connect()`, `context(...)`,
  `event(...) -> (meta, np.ndarray)`, `prebake(...)`, `stats()`, `ping()`. This is the exact algorithm the
  Unity/Unreal clients mirror; see `scripts/sidecar_smoke.py` for an end-to-end run.
- **Unity (C#):** drop-in package at `unity/PredictiveSfx/` (see its README). `PredictiveSfxClient.cs` is
  the low-level transport (reads the 16-byte LE header, loops `Read` until N bytes, `Buffer.BlockCopy` into
  a `float[]`, then `AudioClip.Create` + `SetData`); `PredictiveSfxService` wraps it in a background-thread,
  main-thread-clip-building, auto-reconnecting `MonoBehaviour` with `Emitter`/`Footsteps` components.
- **Unreal (C++):** mirror with `FSocket` `Recv`-until-N for both the header and the body, the same
  `<4sBBBBII>` layout, and `USoundWaveProcedural::QueueAudio` for playback.
