using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

namespace PredictiveSfx
{
    public class PredictiveSfxService : MonoBehaviour
    {
        public string host = "127.0.0.1";
        public int port = 8765;
        public string pcm = "f32le";
        public int reconnectDelayMs = 1000;
        public int keepaliveSeconds = 5;
        public int maxCachedClips = 256;

        static PredictiveSfxService _instance;

        public static PredictiveSfxService Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = FindObjectOfType<PredictiveSfxService>();
                    if (_instance == null)
                    {
                        var go = new GameObject("PredictiveSfxService");
                        _instance = go.AddComponent<PredictiveSfxService>();
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
            public ContextArgs ctx;
            public List<SfxPrediction> predictions;
            public float predictNow;
            public string eventType, material, intensity, context;
            public Action<AudioClip> onReady;
            public Action onFailed;
        }

        class Result
        {
            public bool ok;
            public WorkItem item;
            public float[] samples;
            public int sampleRate, variantId;
            public string served;
        }

        readonly BlockingCollection<WorkItem> _queue = new BlockingCollection<WorkItem>();
        readonly ConcurrentQueue<Result> _results = new ConcurrentQueue<Result>();
        readonly Dictionary<string, LinkedListNode<KeyValuePair<string, AudioClip>>> _clipCache = new Dictionary<string, LinkedListNode<KeyValuePair<string, AudioClip>>>();
        readonly LinkedList<KeyValuePair<string, AudioClip>> _lru = new LinkedList<KeyValuePair<string, AudioClip>>();
        readonly ManualResetEventSlim _stop = new ManualResetEventSlim(false);
        readonly object _clientLock = new object();

        PredictiveSfxClient _client;
        Thread _worker;
        volatile bool _running;
        volatile bool _connected;
        volatile ContextArgs _lastContext;

        public bool IsConnected => _connected;
        public string LastServed { get; private set; }
        public int LastVariantId { get; private set; } = -1;
        public int CachedCount { get; private set; }
        public int FallbackCount { get; private set; }
        public int PredictionsSent { get; private set; }
        public string LastPredicted { get; private set; }

        void Awake()
        {
            if (_instance != null && _instance != this) { Destroy(gameObject); return; }
            _instance = this;
            DontDestroyOnLoad(gameObject);
            _running = true;
            _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "PredictiveSfxWorker" };
            _worker.Start();
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
            if (_running) _queue.Add(new WorkItem { isContext = true, ctx = ctx });
        }

        public void SendPredictions(List<SfxPrediction> preds, float now)
        {
            if (!_running || preds == null || preds.Count == 0) return;
            var p = preds[0];
            LastPredicted = p.eventType + "|" + p.material + " in " +
                p.etaS.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) + "s";
            PredictionsSent += preds.Count;
            _queue.Add(new WorkItem { isPredict = true, predictions = preds, predictNow = now });
        }

        public void Play(string eventType, string material, AudioSource source, string intensity = "normal")
        {
            Request(eventType, material, intensity, "default",
                clip => { if (source != null) source.PlayOneShot(clip); }, null);
        }

        public void Request(string eventType, string material, string intensity, string context, Action<AudioClip> onReady, Action onFailed)
        {
            if (!_running) { onFailed?.Invoke(); return; }
            _queue.Add(new WorkItem
            {
                isContext = false,
                eventType = eventType,
                material = material,
                intensity = intensity,
                context = context,
                onReady = onReady,
                onFailed = onFailed,
            });
        }

        void Update()
        {
            while (_results.TryDequeue(out var res))
            {
                if (res.ok)
                {
                    LastServed = res.served;
                    LastVariantId = res.variantId;
                    if (res.served == "cached") CachedCount++;
                    else if (res.served == "fallback") FallbackCount++;
                    AudioClip clip = null;
                    try { clip = GetOrCreateClip(res); }
                    catch (Exception) { clip = null; }
                    if (clip != null) res.item.onReady?.Invoke(clip);
                    else res.item.onFailed?.Invoke();
                }
                else
                {
                    res.item.onFailed?.Invoke();
                }
            }
        }

        AudioClip GetOrCreateClip(Result res)
        {
            if (res.samples == null || res.samples.Length == 0) return null;
            var it = res.item;
            string key = it.eventType + "|" + it.material + "|" + it.intensity + "|" + res.variantId;
            if (_clipCache.TryGetValue(key, out var node))
            {
                if (node.Value.Value != null)
                {
                    _lru.Remove(node);
                    _lru.AddLast(node);
                    return node.Value.Value;
                }
                _lru.Remove(node);
                _clipCache.Remove(key);
            }
            var clip = AudioClip.Create(key, res.samples.Length, 1, res.sampleRate, false);
            clip.SetData(res.samples, 0);
            _clipCache[key] = _lru.AddLast(new KeyValuePair<string, AudioClip>(key, clip));
            if (_clipCache.Count > Math.Max(1, maxCachedClips))
            {
                var oldest = _lru.First;
                _lru.RemoveFirst();
                _clipCache.Remove(oldest.Value.Key);
                if (oldest.Value.Value != null) Destroy(oldest.Value.Value);
            }
            return clip;
        }

        void DestroyAllClips()
        {
            foreach (var kv in _lru) if (kv.Value != null) Destroy(kv.Value);
            _lru.Clear();
            _clipCache.Clear();
        }

        void WorkerLoop()
        {
            while (_running)
            {
                WorkItem item = null;
                bool got = false;
                try { got = _queue.TryTake(out item, Math.Max(1, keepaliveSeconds) * 1000); }
                catch { break; }
                if (!_running) break;
                try
                {
                    var client = EnsureConnected();
                    if (!got) { client.Ping(0f); continue; }
                    if (item.isPredict)
                    {
                        client.Predict(item.predictions, item.predictNow);
                    }
                    else if (item.isContext)
                    {
                        var c = item.ctx;
                        client.Context(c.timeS, c.equipped, c.surfaces, c.region, c.gpu);
                    }
                    else
                    {
                        var r = client.Event(item.eventType, item.material, item.intensity, item.context);
                        _results.Enqueue(new Result { ok = true, item = item, samples = r.samples, sampleRate = r.meta.sample_rate, variantId = r.meta.variant_id, served = r.meta.served });
                    }
                }
                catch (SfxProtocolException pe)
                {
                    if (got && item != null && !item.isContext && !item.isPredict) _results.Enqueue(new Result { ok = false, item = item });
                    if (pe.Fatal) { _connected = false; SafeClose(); _stop.Wait(reconnectDelayMs); }
                }
                catch (Exception)
                {
                    _connected = false;
                    SafeClose();
                    if (got && item != null && !item.isContext && !item.isPredict) _results.Enqueue(new Result { ok = false, item = item });
                    _stop.Wait(reconnectDelayMs);
                }
            }
            SafeClose();
        }

        PredictiveSfxClient EnsureConnected()
        {
            if (_connected)
            {
                lock (_clientLock) { if (_client != null) return _client; }
            }
            var c = new PredictiveSfxClient();
            c.Connect(host, port, pcm);
            lock (_clientLock) { _client = c; }
            _connected = true;
            var ctx = _lastContext;
            if (ctx != null) c.Context(ctx.timeS, ctx.equipped, ctx.surfaces, ctx.region, ctx.gpu);
            return c;
        }

        void SafeClose()
        {
            PredictiveSfxClient c;
            lock (_clientLock) { c = _client; _client = null; }
            if (c != null) { try { c.Close(); } catch { } }
        }

        void OnDestroy()
        {
            if (_instance == this) Shutdown();
        }

        void OnApplicationQuit() { Shutdown(); }

        void Shutdown()
        {
            if (!_running) return;
            _running = false;
            _stop.Set();
            try { _queue.CompleteAdding(); } catch { }
            SafeClose();
            if (_worker != null) { try { _worker.Join(1000); } catch { } }
            DestroyAllClips();
        }
    }
}
