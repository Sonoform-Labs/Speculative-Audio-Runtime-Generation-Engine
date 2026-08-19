# Sidecar protocol

The sidecar is a local process that generates sound effects for a game over a socket. The game sends context changes and events, the server answers each event with a PCM clip plus metadata.

Transport is loopback TCP on `127.0.0.1`, default port 8765, everything little-endian. Start the server with `python scripts/run_sidecar.py`, or from Python use `sage.sidecar.serve`. The reference client is `sage.sidecar.SfxClient`.

## Frame layout

Every message is one frame: a 16 byte header, a UTF-8 JSON body, and an optional raw PCM tail on audio replies.

| Offset | Size | Field | Notes |
|-------:|-----:|-------|-------|
| 0  | 4 | magic     | ASCII `PSFX` |
| 4  | 1 | version   | `1` |
| 5  | 1 | msg_type  | see below |
| 6  | 1 | flags     | bit0 set when a PCM tail follows |
| 7  | 1 | reserved  | `0` |
| 8  | 4 | json_len  | uint32 |
| 12 | 4 | pcm_len   | uint32, `0` unless flags bit0 |

`struct` format `<4sBBBBII`. Total frame is `16 + json_len + pcm_len`.

Read exactly N bytes in a loop, since a single `recv` can return short. A `recv` of 0 mid-frame means the peer closed. Validate `json_len <= 65536` and `pcm_len <= 4194304` before allocating.

## Message types

| Opcode | Name | Direction |
|-------:|------|-----------|
| 1  | hello        | client to server |
| 2  | welcome      | server to client |
| 3  | context      | client to server |
| 4  | context_ack  | server to client |
| 5  | event        | client to server |
| 6  | audio        | server to client, carries PCM |
| 7  | prebake      | client to server |
| 8  | prebake_ack  | server to client |
| 9  | stats        | client to server |
| 10 | stats_reply  | server to client |
| 11 | ping         | client to server |
| 12 | pong         | server to client |
| 255| error        | server to client |

## Bodies

```jsonc
{ "type":"hello", "proto":1, "client":"unity", "pcm":"f32le" }
{ "type":"welcome", "proto":1, "server_version":"0.1.0", "generator":"stableaudio3",
  "sample_rate":44100, "channels":1, "pcm":"f32le", "max_pcm_bytes":4194304 }

{ "type":"context", "time_s":0.0, "equipped":"gun", "surfaces":["grass"],
  "region":"field", "gpu_pressure":1.0 }
{ "type":"context_ack", "spec_gen":21, "evicted":0, "cached_keys":3 }

{ "type":"event", "event_type":"shoot", "material":"metal", "intensity":"normal", "context":"default" }
{ "type":"audio", "served":"cached", "variant_id":8, "format":"f32le", "sample_rate":44100,
  "channels":1, "n_samples":26460, "byte_len":105840, "duration_s":0.6,
  "intent":"", "learned":4 }

{ "type":"prebake", "contexts":[ { "time_s":0.0, "equipped":"gun", "surfaces":["grass"], "region":"field" } ] }
{ "type":"prebake_ack", "spec_gen":21, "peak_bytes":2654208, "cached_keys":3 }

{ "type":"stats" }
{ "type":"stats_reply", "stats":{ "spec_gen":21, "hits":24, "fallback":2 }, "wasted":1, "cached_keys":3 }

{ "type":"ping", "t":123.5 }
{ "type":"pong", "t":123.5, "server_t":1750000000.0 }

{ "type":"error", "code":"oversized", "message":"...", "fatal":true }
```

The `served` field is `cached` for a cache hit, `realtime` for a clip generated on the spot, or `fallback` for a bundled clip on a miss.

Error codes `bad_frame`, `bad_version`, and `oversized` are fatal and close the connection. `bad_json`, `bad_fields`, `unknown_type`, and `internal` are not, and the connection stays usable.

## Connection

Connections are persistent and idle-tolerant. The server waits indefinitely for the next request, so a silent client is not dropped. `recv_timeout` only bounds a stall in the middle of a frame. `SO_KEEPALIVE` is on. A periodic ping is a cheap liveness probe and is answered without the engine lock. The cache lives on the server, so it survives client reconnects.

## PCM

The default format `f32le` is raw float32 little-endian mono in `[-1, 1]`, ready for `AudioClip.SetData`. The optional `i16le` halves the bytes: encode `round(x * 32767)`, decode `x / 32767`. Always trust the reply's `format`, `n_samples`, and `sample_rate`.

## Versioning and security

`proto` is a single int, and a mismatch closes the connection. New JSON fields may be added without a bump. There is no auth or encryption, the only gate is the loopback-only bind. Never bind `0.0.0.0`.

## Clients

The Python client is `sage.sidecar.SfxClient`. The Unity client is the package at `unity/Sage`, where `SageClient.cs` is the transport and `SageService` wraps it in a background-thread MonoBehaviour. An Unreal port would mirror the same layout with `FSocket` and `USoundWaveProcedural::QueueAudio`.
