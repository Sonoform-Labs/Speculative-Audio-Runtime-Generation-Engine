from __future__ import annotations

import hashlib
import json
import os
import re
import threading

import numpy as np

from . import writer as wr

VERSION = "sarge-ground-1"
MODES = ("off", "anchors", "words", "caption")
DEFAULT_INDEX = r"G:\sarge\captions\index"
EMBEDDER_DIR = "embedder"
FILES = {"meta": "index.json", "captions": "captions.json", "emb": "emb.npy"}
MAX_TOKENS = 64
POOL = 256
K = {"anchors": 4, "words": 8, "caption": 1}
LEX_EVENT = 0.10
LEX_MAT = 0.04
PEOPLE_PEN = 0.03
MULTI_PEN = 0.04
COUNT_PEN = 0.02
META_PEN = 0.05
NEAR_DUP = 0.98
MIN_SHARE = 2
MAX_SOUND_WORDS = 4
MAX_COMMAS = 2
CHUNK = 65536
SPEECH = re.compile(r"(speak|speaks|speaking|speech|talk|talks|talking|voice|voices|laugh|laughs|laughing|laughter|"
                    r"sing|sings|singing|song|songs|shout|shouts|shouting|yell|yells|yelling|conversation|chatter|"
                    r"chatting|whisper|whispers|whispering|narrat\w*|announc\w*|radio|television|tv)")
PEOPLE = re.compile(r"(man|men|woman|women|person|people|someone|somebody|child|children|kid|kids|boy|boys|girl|"
                    r"girls|baby|babies|crowd|he|she|his|her|him|guy|guys|lady|ladies|adult|adults)")
MULTI = re.compile(r"(while|background|followed|then|before|after|meanwhile|also|accompanied|alongside|together|"
                   r"later|afterwards|eventually|finally)")
COUNT = re.compile(r"(several|multiple|repeatedly|repeated|repeating|many|series|twice|numerous|various|different|"
                   r"continuous|continuously|constant|constantly|rapid|rapidly)")
META = re.compile(r"(generated|generating|synthesized|synthesizer|synthetic|sample|samples|sampled|remix|remixed|"
                  r"effect|effects|variation|variations|played|recorded|recording|recordings|created|creating|"
                  r"designed|processed|edited|version|versions|track|tracks|file|files|library|pack|mix|mixed|"
                  r"loop|looped|looping|identified|simulated|imitated|imitation|mimicked|demonstrated|stereo|mono|"
                  r"binaural|microphone|microphones|studio|atmosphere)")
FILLER = re.compile(r"(heard|hear|hears|sound|sounds|sounding|noise|noises|noisy|making|made|make|makes|being|"
                    r"happening|happens|occurring|occurs|performed|someone|something|somewhere|object|objects|thing|"
                    r"things|their|there|some|close|nearby|time|times|place|area|session|action|activity|various|"
                    r"present|each|other|taken|turned)")
MESSY = re.compile(r"(\"|/|\(|\)|\s-|^-|:|;|\b[A-Z]{3,}\b)")
ANCHOR_HEAD = "captions of real recordings like it:"
ANCHOR_RULE = ("\nThe facts may end with captions that sound libraries gave to real recordings of this kind of sound. "
               "Write the action and the sound the way those captions describe it, and prefer their words for how the "
               "sound is made and how it sounds over words of your own. Never take an object, a person, a place or a "
               "second event from them that the facts do not mention.")
PARAMS = {"pool": POOL, "lex_event": LEX_EVENT, "lex_mat": LEX_MAT, "people_pen": PEOPLE_PEN, "multi_pen": MULTI_PEN,
          "count_pen": COUNT_PEN, "meta_pen": META_PEN, "near_dup": NEAR_DUP, "min_share": MIN_SHARE,
          "max_commas": MAX_COMMAS, "query": "request + request with context, summed, normalised",
          "max_sound_words": MAX_SOUND_WORDS, "max_tokens": MAX_TOKENS, "speech": SPEECH.pattern,
          "people": PEOPLE.pattern, "multi": MULTI.pattern, "count": COUNT.pattern, "meta": META.pattern,
          "filler": FILLER.pattern, "messy": MESSY.pattern, "anchor_head": ANCHOR_HEAD, "anchor_rule": ANCHOR_RULE}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def caption_key(text):
    return " ".join(re.split(r"[^a-z0-9]+", str(text or "").lower())).strip()


class Embedder:
    def __init__(self, folder, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.folder = folder
        self.threads = int(threads)
        self.model_path = os.path.join(folder, "onnx", "model.onnx")
        self.tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self._lock = threading.Lock()

    def ident(self):
        return {"embedder": os.path.basename(os.path.normpath(self.folder)),
                "model": wr.quick_digest(self.model_path),
                "tokenizer": sha256_file(os.path.join(self.folder, "tokenizer.json")), "max_tokens": MAX_TOKENS,
                "pooling": "cls"}

    def encode(self, texts):
        encs = self.tok.encode_batch([str(t) for t in texts])
        n = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), n), dtype=np.int64)
        mask = np.zeros((len(encs), n), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        with self._lock:
            out = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        v = out[:, 0, :].astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def one(self, text):
        return self.encode([text])[0]


class CaptionIndex:
    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, FILES["meta"]), encoding="utf-8") as fh:
            self.meta = json.load(fh)
        with open(os.path.join(folder, FILES["captions"]), encoding="utf-8") as fh:
            caps = json.load(fh)
        self.text = caps["text"]
        self.corpus = caps["corpus"]
        self.codes = caps.get("codes") or {}
        self.emb = np.ascontiguousarray(np.load(os.path.join(folder, FILES["emb"])), dtype=np.float32)
        if self.emb.shape[0] != len(self.text):
            raise ValueError("caption index %s: %d embeddings for %d captions" % (folder, self.emb.shape[0],
                                                                                   len(self.text)))
        self._words = {}

    def digest(self):
        return str(self.meta.get("digest") or "")

    def words(self, i):
        ws = self._words.get(i)
        if ws is None:
            ws = wr._words(self.text[i])
            self._words[i] = ws
        return ws

    def scores(self, q):
        return self.emb @ np.asarray(q, dtype=np.float32)

    def top(self, q, n=POOL):
        s = self.scores(q)
        n = min(int(n), s.shape[0])
        part = np.argpartition(-s, n - 1)[:n]
        r = np.round(s[part].astype(np.float64), 6)
        order = np.lexsort((part, -r))
        return [(int(part[j]), float(r[j])) for j in order]


def facts(text):
    vals = {}
    for line in str(text or "").split("\n"):
        k, _, v = line.partition(": ")
        if v:
            vals[k] = v
    return vals


def request_phrase(text):
    return wr.name_phrase(text)


def context_phrase(text):
    f = facts(text)
    parts = []
    who = re.search(r"\(([^)]*)\)", f.get("made by", ""))
    if who:
        parts.append(who.group(1))
    for k in ("object", "touches"):
        if f.get(k):
            parts.append(f[k])
    act = f.get("action", "")
    if act and act != "use":
        parts.append(act)
    out = ", ".join(p for p in parts if p)
    return " ".join(w for w in out.split() if not wr.CONTROLLER.fullmatch(w.lower().strip(",")))


def query_texts(text):
    req = request_phrase(text) or facts(text).get("event", "")
    ctx = context_phrase(text)
    return req, (req + ", " + ctx if ctx else "")


def _has(words, rx):
    return [w for w in words if rx.fullmatch(w)]


def _content(text, skip=()):
    out = []
    for w in wr._words(text):
        if (len(w) >= 3 and w not in wr.STOP and not wr.CONTROLLER.fullmatch(w)
                and not any(ch.isdigit() for ch in w) and w not in skip and w not in out):
            out.append(w)
    return out


def request_words(text):
    f = facts(text)
    ev = _content(f.get("event", ""))
    return ev, _content(f.get("material", ""), ev)


def _covered(req_words, words):
    if not req_words:
        return 0.0
    return sum(1 for r in req_words if any(wr._near(r, x) for x in words)) / float(len(req_words))


def lexical(req, words):
    return LEX_EVENT * _covered(req[0], words) + LEX_MAT * _covered(req[1], words)


class Grounder:
    def __init__(self, mode, index=None, embedder=None, threads=4, k=None):
        if mode not in MODES or mode == "off":
            raise ValueError("unknown caption grounding %r (choose from %s)" % (mode, ", ".join(MODES[1:])))
        self.mode = mode
        self.folder = index or DEFAULT_INDEX
        self.embedder_dir = embedder or os.path.join(self.folder, EMBEDDER_DIR)
        self.threads = int(threads)
        self.k = int(k) if k else K[mode]
        self.index = None
        self.embedder = None
        self._lock = threading.Lock()
        self._ident = None

    def start(self):
        with self._lock:
            if self.index is None:
                self.index = CaptionIndex(self.folder)
            if self.embedder is None:
                self.embedder = Embedder(self.embedder_dir, self.threads)

    def ident(self):
        if self._ident is None:
            with open(os.path.join(self.folder, FILES["meta"]), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.start()
            self._ident = {"ground": VERSION, "mode": self.mode, "k": self.k, "index": meta.get("digest"),
                           "captions": meta.get("count"), "embedder": self.embedder.ident(), "params": PARAMS}
        return self._ident

    def query(self, text):
        req, full = query_texts(text)
        q = self.embedder.one(req)
        if full:
            q = q + self.embedder.one(full)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    def ranked(self, key, text):
        self.start()
        idx = self.index
        inw = wr._input_words(text)
        req = request_words(text)
        loop = facts(text).get("action") == "loop"
        out = []
        for i, cos in idx.top(self.query(text), POOL):
            ws = idx.words(i)
            if MESSY.search(idx.text[i]) or idx.text[i].count(",") > MAX_COMMAS:
                continue
            if any(w not in inw for w in _has(ws, SPEECH)):
                continue
            if any(w not in inw for w in ws if wr.BANNED_OPEN.fullmatch(w)):
                continue
            pen = PEOPLE_PEN * len([w for w in _has(ws, PEOPLE) if w not in inw])
            pen += MULTI_PEN * len([w for w in _has(ws, MULTI) if w not in inw])
            pen += META_PEN * len([w for w in _has(ws, META) if w not in inw])
            if not loop:
                pen += COUNT_PEN * len([w for w in _has(ws, COUNT) if w not in inw])
            score = round(cos + lexical(req, ws) - pen, 6)
            out.append({"i": i, "cos": cos, "score": score})
        out.sort(key=lambda h: (-h["score"], h["i"]))
        return out

    def hits(self, key, text, n=None):
        n = self.k if n is None else n
        out = []
        for h in self.ranked(key, text):
            if len(out) >= n:
                break
            v = self.index.emb[h["i"]]
            if any(float(self.index.emb[o["i"]] @ v) >= NEAR_DUP for o in out):
                continue
            out.append(h)
        return out

    def info(self, hits):
        idx = self.index
        return {"mode": self.mode, "hits": [[h["i"], idx.corpus[h["i"]], h["cos"], h["score"], idx.text[h["i"]]]
                                            for h in hits]}

    def anchored(self, key, text):
        hits = self.hits(key, text)
        lines = [ANCHOR_HEAD] + ["- " + clean_caption(self.index.text[h["i"]], False) for h in hits]
        return text + "\n" + "\n".join(lines), hits

    def style(self, base):
        st = wr.style_of(base)
        if self.mode == "anchors":
            self.start()
            exs = []
            for user, slots in st["examples"]:
                anchored, _h = self.anchored(None, user)
                exs.append((anchored, slots))
            out = wr._style(st["name"] + "+anchors", st["system"] + ANCHOR_RULE, tuple(exs), st["named"], st["form"],
                            st["banned"], st["collide"])
            out["ex_words"], out["leak"] = st["ex_words"], st["leak"]
            return out
        if self.mode == "words":
            out = wr._style("ground-words", "", (), "source", "title", wr.BANNED_OPEN, True)
            out["optional"] = ("action", "sound")
            return out
        out = wr._style("ground-caption", "", (), "source", "list", wr.BANNED_OPEN, True)
        out["optional"] = ("sound",)
        return out

    def slots(self, key, text, spec):
        head = request_phrase(text) or facts(text).get("event", "")
        src = head[:1].upper() + head[1:]
        ranked = self.ranked(key, text)
        if self.mode == "caption":
            for h in ranked:
                slots = {"source": src, "action": clean_caption(self.index.text[h["i"]]), "sound": ""}
                if not wr.check(slots, text, key, spec):
                    return slots, [h]
            return None, []
        words, used = sound_words(self.index, ranked[:self.k], text)
        slots = {"source": src, "action": "", "sound": ", ".join(words)}
        while words and wr.check(slots, text, key, spec):
            words = words[:-1]
            slots["sound"] = ", ".join(words)
        return slots, used

    def answer(self, key, text, backend, spec):
        if self.mode == "anchors":
            anchored, hits = self.anchored(key, text)
            return backend.complete(wr.messages(anchored, spec)), self.info(hits)
        slots, hits = self.slots(key, text, spec)
        return (json.dumps(slots) if slots else ""), self.info(hits)


def clean_caption(text, lower=True):
    s = " ".join(str(text or "").split()).strip(" .,;:!")
    if lower and s[:1].isupper() and not s[1:2].isupper():
        s = s[:1].lower() + s[1:]
    return s


DESC_SKIP = (SPEECH, PEOPLE, MULTI, COUNT, META, FILLER, wr.BANNED_OPEN, wr.CONTROLLER, wr.COLLIDE)


def descriptive(ws, inw, req):
    out = []
    for w in ws:
        if (len(w) < 3 or w in wr.STOP or w in inw or any(wr._near(w, r) or wr._near(r, w) for r in req)
                or any(ch.isdigit() for ch in w) or any(rx.fullmatch(w) for rx in DESC_SKIP) or w in out):
            continue
        out.append(w)
    return out


def sound_words(index, hits, text):
    inw = wr._input_words(text)
    ev, mat = request_words(text)
    seen, weight, first = {}, {}, {}
    for rank, h in enumerate(hits):
        ws = index.words(h["i"])
        for w in descriptive(ws, inw, ev + mat):
            seen[w] = seen.get(w, 0) + 1
            weight[w] = weight.get(w, 0.0) + h["score"]
            first.setdefault(w, (rank, ws.index(w)))
    keep = [w for w in seen if seen[w] >= MIN_SHARE]
    keep.sort(key=lambda w: (-round(weight[w], 6), first[w]))
    out = []
    for w in keep:
        if any(wr._near(w, x) for x in out):
            continue
        out.append(w)
        if len(out) >= MAX_SOUND_WORDS:
            break
    return out, hits


class GroundBackend:
    name = "ground"

    def __init__(self, grounder):
        self.grounder = grounder
        self.model = "captions"
        self.seen_affinity = None

    def ident(self):
        return {"backend": self.name}

    def start(self):
        self.grounder.start()

    def pause(self):
        pass

    def resume(self):
        pass

    def complete(self, msgs):
        raise RuntimeError("the caption grounder writes without a language model")

    def close(self):
        pass
