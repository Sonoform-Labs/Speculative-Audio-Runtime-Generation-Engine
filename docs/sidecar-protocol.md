# Sidecar protocol

The sidecar is a local process that generates sound effects for a game over a socket. The game sends context changes and events, the server answers each event with a PCM clip plus metadata.

Transport is loopback TCP on `127.0.0.1`, default port 8765, everything little-endian. Start the server with `python scripts/run_sidecar.py`, or from Python use `sarge.sidecar.serve`. The reference client is `sarge.sidecar.SfxClient`.

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
| 13 | predict      | client to server |
| 14 | predict_ack  | server to client |
| 15 | manifest     | client to server |
| 16 | manifest_ack | server to client |
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

Optional fields on an event, a prediction (opcode 13) or a prewarm key, all ignored by older servers:

- `"prompt": "..."` is text the game supplies for that key. It goes to whichever model renders the key (SA3 medium or Woosh) and wins over the server's prompt builder. The server remembers it for the key, so predictions and later events of the same key use it too. Sending a different prompt for a key drops the clips cached for it. It never changes the seed.
- `"class": "weapon"` is an audio-free routing hint, read only under `--route class`, where a class listed in `--gun-class` (default `weapon`) goes to SA3 medium and everything else to Woosh. The class is part of the key there, but the seed ignores it.
- A prediction may send `"event_type": ""`. The server then uses the event that has landed most often on that source's predicted surfaces (a learned kind per `source`, see docs/generic.md), and drops the prediction until one has.
- `"scene": {"src": "npc", "who": "Orc_Brute", "obj": "Club_Wooden", "act": "use", "room": "open"}` describes who made the sound, the object, the action and the room (docs/llm-layer.md). Only `--scene` reads it; the server quantises it into the key, answers with `"scene_key"` and leaves the seed alone. An EVENT that sends `have` can add `"have_scene"` (the `scene_key` of the held takes) and `"have_rev"` (their prompt revision); a reply is meta-only only when both match.
- Under `--writer` a reply carries `"prompt_rev"` once the key's prompt has been upgraded; a client holding older takes of that key should drop them. Under `--woosh-upgrade flow` it moves too: once when a key's takes are swapped for their Flow renders, once when a game prompt replaces the key's prompt, and once when swapped takes leave the server's cache, so the next play of that key drops any Flow take the server no longer holds.
- Under `--woosh-upgrade flow` a `stats_reply` also carries `"upgrade"`: the upgrade state, takes queued and staged, Flow takes rendered, takes swapped and swaps, preemptions, failed renders, keys given up for the session, and the seconds from a key's first hit to its first swap (count, median, max).

A client may send `"trace": true` in its hello. The welcome then carries `"trace": true` and every audio reply on that connection carries a `trace` object describing the path the request took, for the Unity pipeline graph (docs/pipeline-graph.md). A client that leaves the field out gets the replies byte for byte as before, the PCM never changes, and `--no-trace` makes the server refuse the request.

```jsonc
"trace": { "path": "cache", "route": "rest", "gen": "woosh_flow", "prompt": "template", "take": "flow",
           "phys": false, "scene": false, "predicted": true }
```

- `path` is `render` (generated on the spot), `cache` (a held take), `meta` (meta-only, the client holds the take), `sibling`, `fallback` or `silence`.
- `route` is `guns` or `rest` under `--generator hybrid` and empty otherwise. `gen` names the model: `sa3`, `woosh_dflow`, `woosh_flow` for a Flow take, or the generator name (`sa3`, `mock`) without the hybrid.
- `prompt` is `template` (the server's builder), `game` (a prompt the game sent), `writer` (a written prompt) or empty for a generator without prompts.
- `take` is `flow` when the served take is a Flow upgrade, `best` when it is an SA3 best-of-N pick, else empty. `phys` and `scene` say whether a physics bucket or a scene record went into the key, and `predicted` whether a prediction for exactly this key was pending when the event arrived.

A `stats` request may carry `"feed": N`. When the server traces, the reply then adds `"feed"`, the background activity after sequence number N (at most 64 items, oldest first), and `"feed_seq"`, the latest number. Each item has `seq`, `kind` (`render`, `writer`, `writer_swap`, `route_flip`, `flow_take`, `flow_swap`, `sa3_candidate`, `sa3_pick`), `key` as `event|material`, the trace's `route`, `gen`, `prompt`, `phys` and `scene`, and per kind `variant`, `takes`, `why`, `side` or `ms`. A number past the latest one, as after a server restart, starts over from the newest 64 items the server holds. Without `feed` the stats reply is unchanged.

`{"type":"manifest","keys":[{"event_type":"door_creak","material":"wood","scene":{...}}, ...]}` lists keys the game can fire (at most 4096, and the 64 KB JSON limit applies, so send them in chunks). The server queues them for the prompt writer and answers `{"type":"manifest_ack","keys":2,"known":0,"queued":2,"writer":true}`; without a writer it answers with `queued` 0 and `writer` false. Older servers answer with a non-fatal `unknown_type` error.

Error codes `bad_frame`, `bad_version`, and `oversized` are fatal and close the connection. `bad_json`, `bad_fields`, `unknown_type`, and `internal` are not, and the connection stays usable.

## Connection

Connections are persistent and idle-tolerant. The server waits indefinitely for the next request, so a silent client is not dropped. `recv_timeout` only bounds a stall in the middle of a frame. `SO_KEEPALIVE` is on. A periodic ping is a cheap liveness probe and is answered without the engine lock. The cache lives on the server, so it survives client reconnects.

## PCM

The default format `f32le` is raw float32 little-endian mono in `[-1, 1]`, ready for `AudioClip.SetData`. The optional `i16le` halves the bytes: encode `round(x * 32767)`, decode `x / 32767`. Always trust the reply's `format`, `n_samples`, and `sample_rate`.

## Versioning and security

`proto` is a single int, and a mismatch closes the connection. New JSON fields may be added without a bump. There is no auth or encryption, the only gate is the loopback-only bind. Never bind `0.0.0.0`.

## Clients

The Python client is `sarge.sidecar.SfxClient`. The Unity client is the package at `unity/Sarge`, where `SargeClient.cs` is the transport and `SargeService` wraps it in a background-thread MonoBehaviour. An Unreal port would mirror the same layout with `FSocket` and `USoundWaveProcedural::QueueAudio`.
