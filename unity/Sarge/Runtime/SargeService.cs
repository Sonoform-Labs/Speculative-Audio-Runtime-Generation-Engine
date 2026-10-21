using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.SceneManagement;
using Debug = UnityEngine.Debug;

namespace Sarge
{
    public class SargeService : MonoBehaviour
    {
        public string host = "127.0.0.1";
        public int port = 8765;
        public string pcm = "f32le";
        public int reconnectDelayMs = 1000;
        public int keepaliveSeconds = 5;
        public int maxCachedClips = 256;
        public int dspBufferSize = 0;
        public bool logTimings = false;
        public bool resendPredictions = false;
        public string[] persistentPredictionSources = new string[0];
        public bool failFastOffline = false;
        public bool localPlayback = false;
        public int eventConnections = 1;
        public bool controlConnection = false;
        public string context = "default";
        public bool sendScene = false;
        public bool sendManifest = false;
        public bool roomProbe = false;
        public int manifestChunk = 100;

        public static Action<SargeService> Setup;

        static SargeService _instance;
        static readonly Stopwatch _clock = Stopwatch.StartNew();

        public static SargeService Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = FindObjectOfType<SargeService>();
                    if (_instance == null)
                    {
                        var go = new GameObject("SargeService");
                        _instance = go.AddComponent<SargeService>();
                    }
                }
                return _instance;
            }
        }

        class ContextArgs { public float timeS; public string equipped; public string[] surfaces; public string region; public float gpu; }

        class WorkItem
        {
            public bool isContext;
            public bool isPredict;
            public bool isManifest;
            public ContextArgs ctx;
            public List<SfxPrediction> predictions;
            public List<string> manifest;
            public float predictNow;
            public string eventType, material, intensity, context;
            public string eventClass, prompt;
            public string physJson, physSig, havePhys;
            public string sceneJson, haveScene;
            public int haveRev;
            public List<int> have;
            public bool playedLocally;
            public int lane;
            public double tRequest, tDequeue, tReply;
            public Action<AudioClip> onReady;
            public Action onFailed;
        }

        class Result
        {
            public bool ok;
            public WorkItem item;
            public float[] samples;
            public int sampleRate, variantId, channels;
            public string served;
            public string intent;
            public int learned;
            public string physKey;
            public string sceneKey;
            public int promptRev;
            public bool metaOnly;
        }

        class Lane
        {
            public int index;
            public bool control;
            public bool events;
            public readonly BlockingCollection<WorkItem> queue = new BlockingCollection<WorkItem>();
            public SargeClient client;
            public Thread thread;
            public volatile bool connected;
            public int busy;
        }

        class Entry
        {
            public string key;
            public string baseKey;
            public int variant;
            public AudioClip clip;
        }

        struct Pending
        {
            public SfxPrediction p;
            public double dueMs;
        }

        readonly List<Lane> _lanes = new List<Lane>();
        readonly List<Lane> _eventLanes = new List<Lane>();
        Lane _control;
        int _nextLane;
        readonly ConcurrentQueue<Result> _results = new ConcurrentQueue<Result>();
        readonly ConcurrentQueue<KeyValuePair<string, string>> _physLearn = new ConcurrentQueue<KeyValuePair<string, string>>();
        readonly Dictionary<string, LinkedListNode<Entry>> _clipCache = new Dictionary<string, LinkedListNode<Entry>>();
        readonly LinkedList<Entry> _lru = new LinkedList<Entry>();
        readonly Dictionary<string, List<int>> _variants = new Dictionary<string, List<int>>();
        readonly Dictionary<string, int> _rotate = new Dictionary<string, int>();
        readonly Dictionary<string, string> _physMap = new Dictionary<string, string>();
        readonly Dictionary<string, string> _sceneMap = new Dictionary<string, string>();
        readonly Dictionary<string, int> _revs = new Dictionary<string, int>();
        bool _manifestDue = true;
        const int MaxSceneMap = 4096;
        readonly Dictionary<string, List<Pending>> _lastPreds = new Dictionary<string, List<Pending>>();
        readonly object _predLock = new object();
        readonly ManualResetEventSlim _stop = new ManualResetEventSlim(false);
        readonly object _clientLock = new object();

        volatile bool _running;
        volatile ContextArgs _lastContext;
        volatile float _lastNow;
        StreamWriter _timings;

        public bool IsConnected => _lanes.Count > 0 && _lanes[0].connected;
        public string Generator { get; private set; }
        public string LastServed { get; private set; }
        public int LastVariantId { get; private set; } = -1;
        public int CachedCount { get; private set; }
        public int RealtimeCount { get; private set; }
        public int FallbackCount { get; private set; }
        public int LocalCount { get; private set; }
        public int PredictionsSent { get; private set; }
        public string LastPredicted { get; private set; }
        public string LastIntent { get; private set; }
        public string LastPhysKey { get; private set; }
        public string LastSceneKey { get; private set; }
        public int PromptUpgrades { get; private set; }
        public int ManifestKeysSent { get; private set; }
        public int LearnedEvents { get; private set; }

        static double Now() => _clock.Elapsed.TotalMilliseconds;

        void Awake()
        {
            if (_instance != null && _instance != this) { Destroy(gameObject); return; }
            _instance = this;
            DontDestroyOnLoad(gameObject);
            SceneManager.sceneLoaded += OnSceneLoaded;
            Setup?.Invoke(this);
            if (dspBufferSize > 0) ApplyDspBuffer();
            _running = true;
            int ev = Math.Max(1, eventConnections);
            for (int i = 0; i < ev; i++)
            {
                var lane = new Lane { index = i, events = true, control = i == 0 && !controlConnection };
                _lanes.Add(lane);
                _eventLanes.Add(lane);
            }
            if (controlConnection) _lanes.Add(new Lane { index = ev, events = false, control = true });
            foreach (var lane in _lanes) if (lane.control) _control = lane;
            foreach (var lane in _lanes)
            {
                var l = lane;
                l.thread = new Thread(() => WorkerLoop(l)) { IsBackground = true, Name = l.index == 0 ? "SargeWorker" : "SargeWorker" + l.index };
                l.thread.Start();
            }
        }

        void ApplyDspBuffer()
        {
            var cfg = AudioSettings.GetConfiguration();
            if (cfg.dspBufferSize != dspBufferSize)
            {
                cfg.dspBufferSize = dspBufferSize;
                if (!AudioSettings.Reset(cfg)) Debug.LogWarning("[SARGE] could not set the DSP buffer to " + dspBufferSize);
            }
            AudioSettings.GetDSPBufferSize(out int len, out int num);
            if (len > 512)
                Debug.LogWarning("[SARGE] DSP buffer is " + len + " x " + num + " samples, about " +
                    (1000f * len * num / Mathf.Max(1, AudioSettings.outputSampleRate)).ToString("0", CultureInfo.InvariantCulture) +
                    " ms of output lag; 256 or 512 plays sounds sooner");
        }

        public void Configure(string newHost, int newPort)
        {
            host = newHost;
            port = newPort;
        }

        public void SetContext(string equipped, string[] surfaces, string region = "default", float gpuPressure = 1f)
        {
            var ctx = new ContextArgs { timeS = Time.time, equipped = equipped, surfaces = surfaces, region = region, gpu = gpuPressure };
            _lastContext = ctx;
            if (_running) Enqueue(_control, new WorkItem { isContext = true, ctx = ctx });
        }

        public void SendPredictions(List<SfxPrediction> preds, float now)
        {
            if (!_running || preds == null || preds.Count == 0) return;
            var p = preds[0];
            LastPredicted = p.eventType + "|" + p.material + " in " +
                p.etaS.ToString("0.00", CultureInfo.InvariantCulture) + "s";
            PredictionsSent += preds.Count;
            if (resendPredictions) Remember(preds);
            Enqueue(_control, new WorkItem { isPredict = true, predictions = preds, predictNow = now });
        }

        void Enqueue(Lane lane, WorkItem item)
        {
            Interlocked.Increment(ref lane.busy);
            lane.queue.Add(item);
        }

        void Remember(List<SfxPrediction> preds)
        {
            double t = Now();
            var slots = new Dictionary<string, List<Pending>>();
            foreach (var p in preds)
            {
                string src = p.source ?? "motion";
                string slot = Persistent(src) ? src
                    : src + "|" + p.eventType + "|" + p.material + "|" + (p.context ?? "default") + "|" + (p.physSig ?? "");
                if (!slots.TryGetValue(slot, out var list)) slots[slot] = list = new List<Pending>();
                list.Add(new Pending { p = p, dueMs = t + p.etaS * 1000.0 });
            }
            lock (_predLock)
            {
                List<string> stale = null;
                foreach (var kv in _lastPreds)
                {
                    bool live = false;
                    foreach (var e in kv.Value) if (Persistent(e.p.source) || e.dueMs > t) { live = true; break; }
                    if (!live) (stale ??= new List<string>()).Add(kv.Key);
                }
                if (stale != null) foreach (var k in stale) _lastPreds.Remove(k);
                foreach (var kv in slots) _lastPreds[kv.Key] = kv.Value;
            }
        }

        bool Persistent(string source)
        {
            if (persistentPredictionSources == null) return false;
            foreach (var s in persistentPredictionSources) if (s == (source ?? "motion")) return true;
            return false;
        }

        void Resend(SargeClient c)
        {
            List<SfxPrediction> preds = null;
            double t = Now();
            lock (_predLock)
            {
                foreach (var kv in _lastPreds)
                    foreach (var e in kv.Value)
                    {
                        bool keep = Persistent(e.p.source);
                        if (!keep && e.dueMs <= t) continue;
                        var p = e.p;
                        if (!keep) p.etaS = Mathf.Max(0.05f, (float)((e.dueMs - t) / 1000.0));
                        (preds ??= new List<SfxPrediction>()).Add(p);
                    }
            }
            if (preds != null) Learn(preds, c.Predict(preds, _lastNow));
        }

        void Learn(List<SfxPrediction> preds, PredictAck ack)
        {
            if (ack == null || ack.phys_keys == null) return;
            for (int i = 0; i < preds.Count && i < ack.phys_keys.Length; i++)
                if (!string.IsNullOrEmpty(preds[i].physSig) && !string.IsNullOrEmpty(ack.phys_keys[i]))
                    _physLearn.Enqueue(new KeyValuePair<string, string>(preds[i].physSig + "|" + (preds[i].intensity ?? "normal"), ack.phys_keys[i]));
        }

        public string ContextOr(string own)
        {
            if (!string.IsNullOrEmpty(own)) return own;
            return string.IsNullOrEmpty(context) ? "default" : context;
        }

        public void Play(string eventType, string material, AudioSource source, string intensity = "normal")
        {
            Request(eventType, material, intensity, ContextOr(null),
                clip => { if (source != null) source.PlayOneShot(clip); }, null);
        }

        public void Request(string eventType, string material, string intensity, string context, Action<AudioClip> onReady, Action onFailed)
        {
            Request(eventType, material, intensity, context, null, null, onReady, onFailed);
        }

        public void Request(string eventType, string material, string intensity, string context, string physJson, string physSig,
            Action<AudioClip> onReady, Action onFailed, string eventClass = null, string prompt = null, string scene = null)
        {
            if (!_running) { onFailed?.Invoke(); return; }
            var item = new WorkItem
            {
                isContext = false,
                eventType = eventType,
                material = material ?? "",
                intensity = intensity,
                context = ContextOr(context),
                eventClass = eventClass,
                prompt = prompt,
                physJson = physJson,
                physSig = physSig,
                sceneJson = string.IsNullOrEmpty(scene) ? null : scene,
                onReady = onReady,
                onFailed = onFailed,
            };
            if (logTimings) item.tRequest = Now();
            var lane = PickLane();
            item.lane = lane.index;
            if (localPlayback) PlayLocal(item);
            if (failFastOffline && !lane.connected) { _results.Enqueue(new Result { ok = false, item = item }); return; }
            Enqueue(lane, item);
        }

        Lane PickLane()
        {
            if (_eventLanes.Count == 1) return _eventLanes[0];
            Lane best = null;
            for (int k = 0; k < _eventLanes.Count; k++)
            {
                var l = _eventLanes[(_nextLane + k) % _eventLanes.Count];
                if (best == null || (l.connected && !best.connected) ||
                    (l.connected == best.connected && Volatile.Read(ref l.busy) < Volatile.Read(ref best.busy))) best = l;
            }
            _nextLane = (best.index + 1) % _eventLanes.Count;
            return best;
        }

        static string Suffix(string context, string physKey, string sceneKey = null)
        {
            string s = string.IsNullOrEmpty(context) || context == "default" ? "" : "|" + context;
            s = string.IsNullOrEmpty(physKey) ? s : s + "|" + physKey;
            return string.IsNullOrEmpty(sceneKey) ? s : s + "|scene=" + sceneKey;
        }

        static string BaseKey(string e, string m, string i, string context, string physKey, string sceneKey = null)
        {
            return e + "|" + m + "|" + i + Suffix(context, physKey, sceneKey);
        }

        static string ClipKey(string e, string m, string i, string context, string physKey, int variant, string sceneKey = null)
        {
            return e + "|" + m + "|" + i + "|" + variant + Suffix(context, physKey, sceneKey);
        }

        void PlayLocal(WorkItem item)
        {
            string physKey = "";
            if (!string.IsNullOrEmpty(item.physJson) &&
                (string.IsNullOrEmpty(item.physSig) || !_physMap.TryGetValue(item.physSig + "|" + item.intensity, out physKey)))
                return;
            string sceneKey = "";
            if (item.sceneJson != null && !_sceneMap.TryGetValue(item.sceneJson, out sceneKey)) return;
            string baseKey = BaseKey(item.eventType, item.material, item.intensity, item.context, physKey, sceneKey);
            if (!_variants.TryGetValue(baseKey, out var ids) || ids.Count == 0) return;
            _rotate.TryGetValue(baseKey, out int next);
            int variant = ids[next % ids.Count];
            _rotate[baseKey] = next + 1;
            if (!_clipCache.TryGetValue(ClipKey(item.eventType, item.material, item.intensity, item.context, physKey, variant, sceneKey), out var node) ||
                node.Value.clip == null)
                return;
            _lru.Remove(node);
            _lru.AddLast(node);
            item.have = new List<int>(ids);
            item.havePhys = physKey;
            item.haveScene = sceneKey;
            _revs.TryGetValue(baseKey, out item.haveRev);
            item.playedLocally = true;
            LocalCount++;
            LastServed = "local";
            LastVariantId = variant;
            if (logTimings) LogRow(item, "local", variant, node.Value.clip);
            item.onReady?.Invoke(node.Value.clip);
        }

        void Update()
        {
            _lastNow = Time.time;
            if (roomProbe) SargeScene.Tick(Time.unscaledTime);
            if (_manifestDue && sendManifest && _running)
            {
                _manifestDue = false;
                SendManifest();
            }
            Drain();
        }

        void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            SargeScene.Clear();
            _manifestDue = true;
        }

        public void SendManifest()
        {
            var keys = new List<string>();
            var seen = new HashSet<string>();
            foreach (var e in FindObjectsOfType<SargeEmitter>())
            {
                if (string.IsNullOrEmpty(e.eventType)) continue;
                Add(keys, seen, ManifestKey(e.eventType, e.material, e.intensity, ContextOr(e.context), e.eventClass, e.prompt,
                    sendScene ? SargeScene.ForEmitter(e, e.madeBy) : null));
            }
            var surfaces = new List<string>();
            foreach (var s in FindObjectsOfType<SargeSurface>())
                if (!string.IsNullOrEmpty(s.material) && !surfaces.Contains(s.material)) surfaces.Add(s.material);
            foreach (var f in FindObjectsOfType<SargeFootsteps>())
            {
                string scene = sendScene ? SargeScene.ForActor(f, "step") : null;
                foreach (var m in surfaces) Add(keys, seen, ManifestKey(f.eventType, m, "normal", ContextOr(f.context), null, null, scene));
                if (!string.IsNullOrEmpty(f.defaultMaterial))
                    Add(keys, seen, ManifestKey(f.eventType, f.defaultMaterial, "normal", ContextOr(f.context), null, null, scene));
            }
            int chunk = Math.Max(1, manifestChunk);
            for (int i = 0; i < keys.Count; i += chunk)
                Enqueue(_control, new WorkItem { isManifest = true, manifest = keys.GetRange(i, Math.Min(chunk, keys.Count - i)) });
            ManifestKeysSent += keys.Count;
        }

        static void Add(List<string> keys, HashSet<string> seen, string key)
        {
            if (seen.Add(key)) keys.Add(key);
        }

        public static string ManifestKey(string eventType, string material, string intensity, string context, string eventClass,
            string prompt, string scene)
        {
            var sb = new StringBuilder(200);
            sb.Append("{\"event_type\":").Append(Json(eventType ?? ""));
            sb.Append(",\"material\":").Append(Json(material ?? ""));
            sb.Append(",\"intensity\":").Append(Json(string.IsNullOrEmpty(intensity) ? "normal" : intensity));
            sb.Append(",\"context\":").Append(Json(string.IsNullOrEmpty(context) ? "default" : context));
            if (!string.IsNullOrEmpty(eventClass)) sb.Append(",\"class\":").Append(Json(eventClass));
            if (!string.IsNullOrEmpty(prompt)) sb.Append(",\"prompt\":").Append(Json(prompt));
            if (!string.IsNullOrEmpty(scene)) sb.Append(",\"scene\":").Append(scene);
            return sb.Append('}').ToString();
        }

        static string Json(string s)
        {
            var sb = new StringBuilder(s.Length + 2);
            sb.Append('"');
            foreach (char ch in s)
            {
                if (ch == '"') sb.Append("\\\"");
                else if (ch == '\\') sb.Append("\\\\");
                else if (ch < ' ') sb.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                else sb.Append(ch);
            }
            return sb.Append('"').ToString();
        }

        void LateUpdate()
        {
            if (localPlayback) Drain();
        }

        void Drain()
        {
            while (_physLearn.TryDequeue(out var kv)) _physMap[kv.Key] = kv.Value;
            while (_results.TryDequeue(out var res))
            {
                var it = res.item;
                if (res.ok)
                {
                    LastServed = it.playedLocally ? "local" : res.served;
                    LastVariantId = res.variantId;
                    if (!string.IsNullOrEmpty(res.intent)) LastIntent = res.intent;
                    if (res.learned > 0) LearnedEvents = res.learned;
                    if (!string.IsNullOrEmpty(res.physKey))
                    {
                        LastPhysKey = res.physKey;
                        if (!string.IsNullOrEmpty(it.physSig)) _physMap[it.physSig + "|" + it.intensity] = res.physKey;
                    }
                    if (it.sceneJson != null)
                    {
                        if (_sceneMap.Count >= MaxSceneMap) _sceneMap.Clear();
                        _sceneMap[it.sceneJson] = res.sceneKey ?? "";
                        if (!string.IsNullOrEmpty(res.sceneKey)) LastSceneKey = res.sceneKey;
                    }
                    if (res.promptRev > 0 || _revs.Count > 0)
                    {
                        string bk = BaseKey(it.eventType, it.material, it.intensity, it.context, res.physKey, res.sceneKey);
                        _revs.TryGetValue(bk, out int known);
                        if (res.promptRev != known)
                        {
                            if (res.promptRev > known && (known > 0 || _variants.ContainsKey(bk))) PromptUpgrades++;
                            DropBase(bk);
                            if (res.promptRev > 0) _revs[bk] = res.promptRev;
                            else _revs.Remove(bk);
                        }
                    }
                    if (res.served == "cached") CachedCount++;
                    else if (res.served == "realtime") RealtimeCount++;
                    else if (res.served == "fallback") FallbackCount++;
                    AudioClip clip = null;
                    if (!res.metaOnly)
                    {
                        try { clip = res.served == "cached" || res.served == "realtime" ? GetOrCreateClip(res) : TransientClip(res); }
                        catch (Exception) { clip = null; }
                    }
                    if (it.playedLocally) continue;
                    if (logTimings) LogRow(it, res.served, res.variantId, clip);
                    if (clip != null) it.onReady?.Invoke(clip);
                    else it.onFailed?.Invoke();
                }
                else
                {
                    if (it.playedLocally) continue;
                    if (logTimings) LogRow(it, "failed", -1, null);
                    it.onFailed?.Invoke();
                }
            }
        }

        AudioClip GetOrCreateClip(Result res)
        {
            if (res.samples == null || res.samples.Length == 0) return null;
            var it = res.item;
            string key = ClipKey(it.eventType, it.material, it.intensity, it.context, res.physKey, res.variantId, res.sceneKey);
            if (_clipCache.TryGetValue(key, out var node))
            {
                if (node.Value.clip != null)
                {
                    _lru.Remove(node);
                    _lru.AddLast(node);
                    return node.Value.clip;
                }
                _lru.Remove(node);
                _clipCache.Remove(key);
                Unlist(node.Value);
            }
            int ch = Math.Max(1, res.channels);
            var clip = AudioClip.Create(key, res.samples.Length / ch, ch, res.sampleRate, false);
            clip.SetData(res.samples, 0);
            var entry = new Entry
            {
                key = key,
                baseKey = BaseKey(it.eventType, it.material, it.intensity, it.context, res.physKey, res.sceneKey),
                variant = res.variantId,
                clip = clip,
            };
            _clipCache[key] = _lru.AddLast(entry);
            if (!_variants.TryGetValue(entry.baseKey, out var ids)) _variants[entry.baseKey] = ids = new List<int>();
            if (!ids.Contains(entry.variant)) ids.Add(entry.variant);
            if (_clipCache.Count > Math.Max(1, maxCachedClips))
            {
                var oldest = _lru.First;
                _lru.RemoveFirst();
                _clipCache.Remove(oldest.Value.key);
                Unlist(oldest.Value);
                if (oldest.Value.clip != null) Destroy(oldest.Value.clip);
            }
            return clip;
        }

        AudioClip TransientClip(Result res)
        {
            if (res.samples == null || res.samples.Length == 0) return null;
            var it = res.item;
            int ch = Math.Max(1, res.channels);
            var clip = AudioClip.Create(ClipKey(it.eventType, it.material, it.intensity, it.context, res.physKey, res.variantId, res.sceneKey) + "|" + res.served,
                res.samples.Length / ch, ch, res.sampleRate, false);
            clip.SetData(res.samples, 0);
            Destroy(clip, clip.length + 5f);
            return clip;
        }

        void DropBase(string baseKey)
        {
            var node = _lru.First;
            while (node != null)
            {
                var next = node.Next;
                if (node.Value.baseKey == baseKey)
                {
                    _lru.Remove(node);
                    _clipCache.Remove(node.Value.key);
                    if (node.Value.clip != null) Destroy(node.Value.clip);
                }
                node = next;
            }
            _variants.Remove(baseKey);
            _rotate.Remove(baseKey);
        }

        void Unlist(Entry e)
        {
            if (_variants.TryGetValue(e.baseKey, out var ids))
            {
                ids.Remove(e.variant);
                if (ids.Count == 0) _variants.Remove(e.baseKey);
            }
        }

        void DestroyAllClips()
        {
            foreach (var e in _lru) if (e.clip != null) Destroy(e.clip);
            _lru.Clear();
            _clipCache.Clear();
            _variants.Clear();
        }

        void LogRow(WorkItem it, string served, int variant, AudioClip clip)
        {
            try
            {
                if (_timings == null)
                {
                    AudioSettings.GetDSPBufferSize(out int len, out int num);
                    string path = Path.Combine(Application.persistentDataPath,
                        "sarge_timings_" + DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture) + ".csv");
                    _timings = new StreamWriter(path, false);
                    _timings.WriteLine("# dsp_buffer=" + len + " x" + num + " output_rate=" + AudioSettings.outputSampleRate +
                        " lanes=" + _lanes.Count + " local=" + localPlayback);
                    _timings.WriteLine("request_ms,dequeue_ms,reply_ms,play_ms,dsp_time,frame,served,lane,variant,samples,key");
                    Debug.Log("[SARGE] timing log " + path);
                }
                var inv = CultureInfo.InvariantCulture;
                _timings.WriteLine(string.Format(inv, "{0:F3},{1:F3},{2:F3},{3:F3},{4:F5},{5},{6},{7},{8},{9},\"{10}\"",
                    it.tRequest, it.tDequeue, it.tReply, Now(), AudioSettings.dspTime, Time.frameCount, served, it.lane,
                    variant, clip != null ? clip.samples : 0,
                    (BaseKey(it.eventType, it.material, it.intensity, it.context, "") + (string.IsNullOrEmpty(it.physJson) ? "" : "|phys")).Replace("\"", "'")));
                if (Time.frameCount % 30 == 0) _timings.Flush();
            }
            catch (Exception e)
            {
                Debug.LogWarning("[SARGE] timing log failed: " + e.Message);
                logTimings = false;
            }
        }

        void WorkerLoop(Lane lane)
        {
            bool eager = lane.index > 0 || controlConnection;
            while (_running)
            {
                WorkItem item = null;
                bool got = false;
                int wait = (eager || failFastOffline) && !lane.connected ? Math.Max(1, reconnectDelayMs) : Math.Max(1, keepaliveSeconds) * 1000;
                try { got = lane.queue.TryTake(out item, wait); }
                catch { break; }
                if (!_running) break;
                if (got && item != null && logTimings) item.tDequeue = Now();
                try
                {
                    var client = EnsureConnected(lane);
                    if (!got) { client.Ping(0f); continue; }
                    if (item.isPredict)
                    {
                        Learn(item.predictions, client.Predict(item.predictions, item.predictNow));
                    }
                    else if (item.isContext)
                    {
                        var c = item.ctx;
                        client.Context(c.timeS, c.equipped, c.surfaces, c.region, c.gpu);
                    }
                    else if (item.isManifest)
                    {
                        client.Manifest(item.manifest);
                    }
                    else
                    {
                        var r = client.Event(item.eventType, item.material, item.intensity, item.context, item.physJson, item.have, item.havePhys,
                            item.eventClass, item.prompt, item.sceneJson, item.haveScene, item.haveRev);
                        if (logTimings) item.tReply = Now();
                        _results.Enqueue(new Result { ok = true, item = item, samples = r.samples, sampleRate = r.meta.sample_rate, channels = r.meta.channels, variantId = r.meta.variant_id, served = r.meta.served, intent = r.meta.intent, learned = r.meta.learned, physKey = r.meta.phys_key, sceneKey = r.meta.scene_key, promptRev = r.meta.prompt_rev, metaOnly = r.meta.meta_only });
                    }
                }
                catch (SfxProtocolException pe)
                {
                    if (got && item != null && !item.isContext && !item.isPredict && !item.isManifest) _results.Enqueue(new Result { ok = false, item = item });
                    if (pe.Fatal) { lane.connected = false; SafeClose(lane); _stop.Wait(reconnectDelayMs); }
                }
                catch (Exception)
                {
                    lane.connected = false;
                    SafeClose(lane);
                    if (got && item != null && !item.isContext && !item.isPredict && !item.isManifest) _results.Enqueue(new Result { ok = false, item = item });
                    _stop.Wait(reconnectDelayMs);
                }
                finally
                {
                    if (got) Interlocked.Decrement(ref lane.busy);
                }
            }
            SafeClose(lane);
        }

        SargeClient EnsureConnected(Lane lane)
        {
            if (lane.connected)
            {
                lock (_clientLock) { if (lane.client != null) return lane.client; }
            }
            var c = new SargeClient();
            lock (_clientLock) { lane.client = c; }
            try { c.Connect(host, port, pcm); }
            catch
            {
                lock (_clientLock) { if (lane.client == c) lane.client = null; }
                try { c.Close(); } catch { }
                throw;
            }
            if (lane.index == 0) Generator = c.Welcome != null ? c.Welcome.generator : null;
            lock (_clientLock) { lane.client = c; }
            lane.connected = true;
            if (lane.control)
            {
                var ctx = _lastContext;
                if (ctx != null) c.Context(ctx.timeS, ctx.equipped, ctx.surfaces, ctx.region, ctx.gpu);
                if (resendPredictions) Resend(c);
            }
            return c;
        }

        void SafeClose(Lane lane)
        {
            SargeClient c;
            lock (_clientLock) { c = lane.client; lane.client = null; }
            if (c != null) { try { c.Close(); } catch { } }
        }

        void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            if (_instance == this) Shutdown();
        }

        void OnApplicationQuit() { Shutdown(); }

        void Shutdown()
        {
            if (!_running) return;
            _running = false;
            _stop.Set();
            foreach (var lane in _lanes) { try { lane.queue.CompleteAdding(); } catch { } }
            foreach (var lane in _lanes) SafeClose(lane);
            foreach (var lane in _lanes) if (lane.thread != null) { try { lane.thread.Join(1000); } catch { } }
            DestroyAllClips();
            if (_timings != null) { try { _timings.Flush(); _timings.Dispose(); } catch { } _timings = null; }
        }
    }
}
