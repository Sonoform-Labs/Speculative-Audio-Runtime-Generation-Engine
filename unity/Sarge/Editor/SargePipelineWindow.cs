using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEditor;
using UnityEngine;

namespace Sarge
{
    public class SargePipelineWindow : EditorWindow, IHasCustomMenu
    {
        enum N { GameEvent, WorldLayer, Predictors, WorldModel, Writer, Prompt, Router, Sa3, DFlow, Shaping, Cache, Flow, Selection, Playback }

        enum Shape { Side, Rise, Up, Arch, Sag, Loop }

        const float NodeW = 160f, NodeH = 72f, HeaderH = 22f, PinY = 47f, Pitch = 200f;
        const float Hold = 0.22f, Fade = 1.1f, Attack = 0.06f, Stagger = 0.075f;
        const string ToolbarPref = "Sarge.PipelineGraph.Toolbar";
        static readonly Rect Bounds = new Rect(-10f, -110f, 1810f, 640f);
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        static readonly Color Bg = new Color(0.150f, 0.150f, 0.150f);
        static readonly Color MinorLine = new Color(0.185f, 0.185f, 0.185f);
        static readonly Color MajorLine = new Color(0.085f, 0.085f, 0.085f);
        static readonly Color NodeBody = new Color(0.055f, 0.055f, 0.06f, 0.94f);
        static readonly Color CaptionIdle = new Color(0.78f, 0.78f, 0.80f);

        class Pulse
        {
            public double start;
            public float strength;
        }

        class Node
        {
            public N id;
            public string title, role;
            public Color color;
            public Vector2 pos;
            public string caption = "";
            public string pending;
            public double pendingAt = double.MaxValue;
            public readonly List<Pulse> pulses = new List<Pulse>();
        }

        class Wire
        {
            public N a, b;
            public Shape shape;
            public float depth;
            public bool dashed;
            public readonly List<Pulse> pulses = new List<Pulse>();
        }

        struct Step
        {
            public N n;
            public string cap;
            public float k;

            public Step(N n, string cap, float k = 1f)
            {
                this.n = n;
                this.cap = cap;
                this.k = k;
            }
        }

        Node[] _nodes;
        List<Wire> _wires;
        readonly List<SargeTraceRecord> _incoming = new List<SargeTraceRecord>();
        readonly List<Step> _steps = new List<Step>();
        long _seq;
        double _lastPaint;
        bool _auto = true;
        float _zoom = 1f;
        Vector2 _offset;
        Texture2D _grad;
        GUIStyle _title, _role, _caption;
        int _fontKey = -1;
        int _preview;

        [MenuItem("Window/SARGE/Pipeline Graph")]
        public static void Open()
        {
            var w = GetWindow<SargePipelineWindow>();
            w.titleContent = new GUIContent("SARGE Pipeline");
            w.minSize = new Vector2(360f, 180f);
            w.Show();
        }

        void OnEnable()
        {
            titleContent = new GUIContent("SARGE Pipeline");
            Build();
            SargeTrace.Listen();
            _seq = SargeTrace.Latest;
            EditorApplication.update += Tick;
        }

        void OnDisable()
        {
            EditorApplication.update -= Tick;
            SargeTrace.Unlisten();
            if (_grad != null) DestroyImmediate(_grad);
            _grad = null;
        }

        public void AddItemsToMenu(GenericMenu menu)
        {
            menu.AddItem(new GUIContent("Show Toolbar"), ShowToolbar, () => { EditorPrefs.SetBool(ToolbarPref, !ShowToolbar); Repaint(); });
            menu.AddItem(new GUIContent("Frame All"), false, () => { _auto = true; Repaint(); });
            menu.AddItem(new GUIContent("Preview a Path"), false, Preview);
        }

        static bool ShowToolbar => EditorPrefs.GetBool(ToolbarPref, true);

        void Build()
        {
            _nodes = new Node[Enum.GetValues(typeof(N)).Length];
            Add(N.GameEvent, "Game Event", "what the game fired", new Color(0.62f, 0.12f, 0.12f), 0, 300f);
            Add(N.WorldLayer, "World Layer", "who, what, material, room", new Color(0.15f, 0.50f, 0.30f), 1, 300f);
            Add(N.Predictors, "Predictors", "motion, aim, impact", new Color(0.82f, 0.48f, 0.08f), 2, 36f);
            Add(N.WorldModel, "World Model", "physics to meaning", new Color(0.05f, 0.48f, 0.54f), 2, 300f);
            Add(N.Writer, "Writer", "small LM writes the prompt", new Color(0.46f, 0.24f, 0.68f), 3, 300f);
            Add(N.Prompt, "Prompt", "template, game or writer", new Color(0.58f, 0.32f, 0.72f), 4, 300f);
            Add(N.Router, "Router", "guns or everything else", new Color(0.42f, 0.42f, 0.44f), 5, 300f);
            Add(N.Sa3, "SA3 Medium", "guns", new Color(0.13f, 0.36f, 0.74f), 6, 236f);
            Add(N.DFlow, "Woosh DFlow", "fast first play", new Color(0.10f, 0.46f, 0.80f), 6, 372f);
            Add(N.Shaping, "Shaping", "trim, loudness, decay, stereo", new Color(0.08f, 0.56f, 0.46f), 7, 300f);
            Add(N.Cache, "Cache", "memory of rendered takes", new Color(0.32f, 0.40f, 0.52f), 7, 140f);
            Add(N.Flow, "Woosh Flow", "background upgrade", new Color(0.05f, 0.55f, 0.82f), 5, 0f);
            Add(N.Selection, "Selection", "listener keeps the best takes", new Color(0.72f, 0.22f, 0.46f), 6, 0f);
            Add(N.Playback, "Playback", "Unity audio source", new Color(0.78f, 0.32f, 0.10f), 8, 220f);
            _wires = new List<Wire>();
            Link(N.GameEvent, N.WorldLayer);
            Link(N.WorldLayer, N.Predictors, Shape.Rise);
            Link(N.WorldLayer, N.Cache, Shape.Rise);
            Link(N.Predictors, N.Cache);
            Link(N.WorldLayer, N.WorldModel);
            Link(N.WorldModel, N.Writer);
            Link(N.Writer, N.Prompt);
            Link(N.WorldLayer, N.Writer, Shape.Sag, 140f);
            Link(N.WorldModel, N.Prompt, Shape.Sag, 140f);
            Link(N.WorldLayer, N.Prompt, Shape.Sag, 210f);
            Link(N.Prompt, N.Router);
            Link(N.Writer, N.Router, Shape.Arch, 80f);
            Link(N.Router, N.Sa3);
            Link(N.Router, N.DFlow);
            Link(N.Sa3, N.Shaping);
            Link(N.DFlow, N.Shaping);
            Link(N.Shaping, N.Playback);
            Link(N.Cache, N.Playback);
            Link(N.Shaping, N.Cache, Shape.Up);
            Link(N.Cache, N.Flow, Shape.Loop, 120f, true);
            Link(N.Flow, N.Selection);
            Link(N.Selection, N.Cache);
            Link(N.Sa3, N.Selection, Shape.Up);
        }

        void Add(N id, string title, string role, Color c, int col, float y)
        {
            _nodes[(int)id] = new Node { id = id, title = title, role = role, color = c, pos = new Vector2(20f + Pitch * col, y) };
        }

        void Link(N a, N b, Shape s = Shape.Side, float depth = 0f, bool dashed = false)
        {
            _wires.Add(new Wire { a = a, b = b, shape = s, depth = depth, dashed = dashed });
        }

        Wire Find(N a, N b)
        {
            foreach (var w in _wires) if (w.a == a && w.b == b) return w;
            return null;
        }

        void Tick()
        {
            if (_nodes == null) Build();
            double now = EditorApplication.timeSinceStartup;
            Pull(now);
            double gap = Hot(now) ? 1.0 / 60.0 : EditorApplication.isPlaying ? 0.25 : 1.0;
            if (now - _lastPaint < gap) return;
            _lastPaint = now;
            Repaint();
        }

        void Pull(double now)
        {
            _incoming.Clear();
            long latest = SargeTrace.Since(_seq, _incoming);
            if (latest < _seq)
            {
                _incoming.Clear();
                latest = SargeTrace.Since(0, _incoming);
            }
            _seq = latest;
            int n = _incoming.Count;
            for (int i = Math.Max(0, n - 48); i < n; i++) Apply(_incoming[i], now);
        }

        bool Hot(double now)
        {
            foreach (var n in _nodes)
            {
                if (n.pending != null) return true;
                foreach (var p in n.pulses) if (now - p.start <= Hold + Fade) return true;
            }
            foreach (var w in _wires)
                foreach (var p in w.pulses) if (now - p.start <= Hold + Fade) return true;
            return false;
        }

        void Apply(SargeTraceRecord r, double now)
        {
            string key = Key(r);
            var s = Steps();
            switch (r.kind)
            {
                case "event":
                    OnEventRecord(r, key, now);
                    return;
                case "predict":
                    s.Add(new Step(N.WorldLayer, null, 0.4f));
                    s.Add(new Step(N.Predictors, (r.source ?? "motion") + ", " + r.eta.ToString("0.00", Inv) + " s"));
                    if (r.queued > 0) s.Add(new Step(N.Cache, "queued"));
                    Run(now, 0.75f, s);
                    return;
                case "feed:render":
                    Thinking(r, s);
                    Model(r, s, "background" + Ms(r.ms));
                    s.Add(new Step(N.Shaping, "shaped"));
                    s.Add(new Step(N.Cache, r.detail == "writer_fill" ? "refilled" : "prefetched"));
                    Run(now, 0.75f, s);
                    return;
                case "feed:writer":
                    if (r.phys || r.scene) s.Add(new Step(N.WorldModel, Meaning(r)));
                    s.Add(new Step(N.Writer, "wrote" + Ms(r.ms)));
                    s.Add(new Step(N.Prompt, "ready"));
                    Run(now, 0.85f, s);
                    return;
                case "feed:writer_swap":
                    s.Add(new Step(N.Writer, "re-render"));
                    s.Add(new Step(N.Prompt, "writer"));
                    Model(r, s, Ms(r.ms).Trim());
                    s.Add(new Step(N.Shaping, "shaped"));
                    s.Add(new Step(N.Cache, "swapped"));
                    Run(now, 0.85f, s);
                    return;
                case "feed:route_flip":
                    s.Add(new Step(N.Writer, "gun?"));
                    s.Add(new Step(N.Router, "by meaning"));
                    Run(now, 0.85f, s);
                    return;
                case "feed:flow_take":
                    s.Add(new Step(N.Cache, "upgrading", 0.6f));
                    s.Add(new Step(N.Flow, "take " + r.variant.ToString(Inv) + (r.ms > 0f ? "," + Ms(r.ms) : "")));
                    s.Add(new Step(N.Selection, "scoring"));
                    Run(now, 0.85f, s);
                    return;
                case "feed:flow_swap":
                    s.Add(new Step(N.Flow, "ready"));
                    s.Add(new Step(N.Selection, "best kept"));
                    s.Add(new Step(N.Cache, "upgraded"));
                    Run(now, 0.95f, s);
                    return;
                case "feed:sa3_candidate":
                    s.Add(new Step(N.Sa3, "take " + r.count.ToString(Inv)));
                    s.Add(new Step(N.Selection, "scoring"));
                    Run(now, 0.75f, s);
                    return;
                case "feed:sa3_pick":
                    s.Add(new Step(N.Sa3, "best of N"));
                    s.Add(new Step(N.Selection, r.count > 0 ? "better take" : "kept"));
                    s.Add(new Step(N.Cache, r.count > 0 ? "swapped" : "kept"));
                    Run(now, 0.85f, s);
                    return;
            }
        }

        void OnEventRecord(SargeTraceRecord r, string key, double now)
        {
            var s = Steps();
            string inten = string.IsNullOrEmpty(r.intensity) || r.intensity == "normal" ? "" : ", " + r.intensity;
            s.Add(new Step(N.GameEvent, key + inten));
            s.Add(new Step(N.WorldLayer, World(r)));
            string path = r.path ?? "";
            if (path == "cache" || path == "meta" || path == "local" || path == "sibling")
            {
                if (r.predicted) s.Add(new Step(N.Predictors, "hit, " + (r.source ?? "predicted")));
                s.Add(new Step(N.Cache, CacheCaption(r)));
                s.Add(new Step(N.Playback, "played"));
                Run(now, 1f, s);
                return;
            }
            if (path == "render")
            {
                Thinking(r, s);
                Model(r, s, Ms(r.ms).Trim());
                s.Add(new Step(N.Shaping, "shaped"));
                s.Add(new Step(N.Playback, "played"));
                Run(now, 1f, s);
                if (r.predicted) Light(N.Predictors, now + Stagger * 2.0, 0.55f, "too late");
                return;
            }
            if (path == "failed")
            {
                s[1] = new Step(N.WorldLayer, "no reply");
                Run(now, 0.8f, s);
                return;
            }
            Run(now, 1f, s);
            Light(N.Playback, now + Stagger * 2.0, 0.8f, path);
        }

        void Thinking(SargeTraceRecord r, List<Step> s)
        {
            if (r.phys || r.scene) s.Add(new Step(N.WorldModel, Meaning(r)));
            if (r.prompt == "writer") s.Add(new Step(N.Writer, "written"));
            s.Add(new Step(N.Prompt, PromptCaption(r.prompt)));
        }

        void Model(SargeTraceRecord r, List<Step> s, string cap)
        {
            N gen = Gen(r.route, r.gen);
            s.Add(new Step(N.Router, RouteCaption(r.route, gen, r.gen)));
            s.Add(new Step(gen, cap));
        }

        List<Step> Steps()
        {
            _steps.Clear();
            return _steps;
        }

        void Run(double now, float strength, List<Step> s)
        {
            for (int i = 0; i < s.Count; i++)
            {
                double at = now + Stagger * i;
                Light(s[i].n, at, strength * s[i].k, s[i].cap);
                if (i == 0) continue;
                var w = Find(s[i - 1].n, s[i].n);
                if (w != null) w.pulses.Add(new Pulse { start = at - Stagger * 0.5, strength = strength * Mathf.Max(s[i].k, s[i - 1].k) });
            }
        }

        void Light(N id, double at, float strength, string cap)
        {
            var n = _nodes[(int)id];
            n.pulses.Add(new Pulse { start = at, strength = strength });
            if (cap == null) return;
            if (n.pending != null && n.pendingAt > at) n.caption = n.pending;
            n.pending = cap;
            n.pendingAt = at;
        }

        static string Key(SargeTraceRecord r)
        {
            string e = r.eventType ?? "";
            if (string.IsNullOrEmpty(r.material)) return e.Length > 0 ? e : "?";
            return (e.Length > 0 ? e : "?") + ", " + r.material;
        }

        static string World(SargeTraceRecord r)
        {
            string m = string.IsNullOrEmpty(r.material) ? "no material" : r.material;
            if (r.physSent) m += ", physics";
            if (r.sceneSent) m += ", scene";
            return m;
        }

        static string Meaning(SargeTraceRecord r)
        {
            return r.phys && r.scene ? "physics, scene" : r.phys ? "physics" : "scene";
        }

        static string CacheCaption(SargeTraceRecord r)
        {
            string where = r.path == "meta" ? "in Unity" : r.path == "local" ? "in Unity"
                : r.path == "sibling" ? "sibling" : "cached";
            string take = r.take == "flow" ? ", Flow" : r.take == "best" ? ", best" : "";
            return where + take + (r.path == "local" ? "" : Ms(r.ms));
        }

        static string PromptCaption(string p)
        {
            if (p == "writer") return "writer";
            if (p == "game") return "game";
            if (p == "template") return "template";
            return "built";
        }

        static string RouteCaption(string route, N gen, string label)
        {
            if (route == "guns") return "guns";
            if (route == "rest") return "Woosh";
            return "one model";
        }

        static N Gen(string route, string gen)
        {
            gen = gen ?? "";
            if (gen == "woosh_flow") return N.Flow;
            if (route == "guns" || gen.StartsWith("sa", StringComparison.Ordinal) || gen.StartsWith("stable", StringComparison.Ordinal)) return N.Sa3;
            if (route == "rest" || gen.StartsWith("woosh", StringComparison.Ordinal)) return N.DFlow;
            return N.Sa3;
        }

        static string Ms(float ms)
        {
            if (ms <= 0f) return "";
            return ms < 10f ? " " + ms.ToString("0.0", Inv) + " ms" : " " + ms.ToString("0", Inv) + " ms";
        }

        void Preview()
        {
            double now = EditorApplication.timeSinceStartup;
            switch (_preview++ % 4)
            {
                case 0:
                    Apply(new SargeTraceRecord { kind = "predict", source = "motion", eventType = "footstep", material = "gravel", eta = 0.6f, queued = 0 }, now);
                    Apply(new SargeTraceRecord { kind = "event", eventType = "footstep", material = "gravel", intensity = "normal", path = "cache",
                        route = "rest", gen = "woosh_dflow", predicted = true, source = "motion", ms = 1.2f }, now + 0.35);
                    break;
                case 1:
                    Apply(new SargeTraceRecord { kind = "event", eventType = "shoot", material = "rifle", intensity = "normal", path = "render",
                        route = "guns", gen = "sa3", prompt = "template", ms = 148f }, now);
                    break;
                case 2:
                    Apply(new SargeTraceRecord { kind = "event", eventType = "glass bottle hit", material = "stone", intensity = "hard", path = "render",
                        route = "rest", gen = "woosh_dflow", prompt = "writer", phys = true, scene = true, physSent = true, sceneSent = true, ms = 31f }, now);
                    break;
                default:
                    Apply(new SargeTraceRecord { kind = "feed:flow_take", eventType = "footstep", material = "gravel", variant = 1, ms = 980f }, now);
                    Apply(new SargeTraceRecord { kind = "feed:flow_swap", eventType = "footstep", material = "gravel", count = 2 }, now + 0.7);
                    break;
            }
            Repaint();
        }

        void OnGUI()
        {
            if (_nodes == null) Build();
            float bar = 0f;
            if (ShowToolbar)
            {
                Toolbar();
                bar = EditorStyles.toolbar.fixedHeight > 0f ? EditorStyles.toolbar.fixedHeight : 21f;
            }
            var canvas = new Rect(0f, bar, position.width, Mathf.Max(1f, position.height - bar));
            Navigate(canvas);
            if (_auto) Fit(canvas);
            Styles();
            if (Event.current.type != EventType.Repaint) return;
            double now = EditorApplication.timeSinceStartup;
            GUI.BeginClip(canvas);
            var local = new Rect(0f, 0f, canvas.width, canvas.height);
            EditorGUI.DrawRect(local, Bg);
            Grid(local);
            foreach (var w in _wires) DrawWire(w, now);
            foreach (var n in _nodes) DrawNode(n, now);
            foreach (var w in _wires) DrawPins(w, now);
            GUI.EndClip();
        }

        void Toolbar()
        {
            GUILayout.BeginHorizontal(EditorStyles.toolbar);
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Preview", EditorStyles.toolbarButton, GUILayout.Width(60f))) Preview();
            if (GUILayout.Button("Frame", EditorStyles.toolbarButton, GUILayout.Width(50f))) _auto = true;
            GUILayout.EndHorizontal();
        }

        void Navigate(Rect canvas)
        {
            var e = Event.current;
            if (e.type != EventType.MouseDrag && e.type != EventType.KeyDown && !canvas.Contains(e.mousePosition)) return;
            Vector2 m = e.mousePosition - canvas.position;
            switch (e.type)
            {
                case EventType.ScrollWheel:
                {
                    float z = Mathf.Clamp(_zoom * (1f - e.delta.y * 0.04f), 0.15f, 3f);
                    _offset = m - (m - _offset) * (z / _zoom);
                    _zoom = z;
                    _auto = false;
                    e.Use();
                    Repaint();
                    break;
                }
                case EventType.MouseDrag:
                    if (e.button == 2 || e.button == 1 || (e.button == 0 && e.alt))
                    {
                        _offset += e.delta;
                        _auto = false;
                        e.Use();
                        Repaint();
                    }
                    break;
                case EventType.KeyDown:
                    if (e.keyCode == KeyCode.F)
                    {
                        _auto = true;
                        e.Use();
                        Repaint();
                    }
                    break;
            }
        }

        void Fit(Rect canvas)
        {
            float z = Mathf.Min((canvas.width - 24f) / Bounds.width, (canvas.height - 24f) / Bounds.height);
            _zoom = Mathf.Clamp(z, 0.15f, 3f);
            _offset = new Vector2((canvas.width - Bounds.width * _zoom) * 0.5f - Bounds.x * _zoom,
                (canvas.height - Bounds.height * _zoom) * 0.5f - Bounds.y * _zoom);
        }

        void Styles()
        {
            int key = Mathf.RoundToInt(_zoom * 100f);
            if (key == _fontKey && _title != null) return;
            _fontKey = key;
            _title = new GUIStyle(EditorStyles.boldLabel) { fontSize = Font(12.5f), alignment = TextAnchor.MiddleLeft, clipping = TextClipping.Clip, wordWrap = false };
            _title.normal.textColor = Color.white;
            _role = new GUIStyle(EditorStyles.label) { fontSize = Font(9.5f), alignment = TextAnchor.MiddleLeft, clipping = TextClipping.Clip, wordWrap = false };
            _role.normal.textColor = new Color(0.60f, 0.60f, 0.63f);
            _caption = new GUIStyle(EditorStyles.boldLabel) { fontSize = Font(10.5f), alignment = TextAnchor.MiddleLeft, clipping = TextClipping.Clip, wordWrap = false };
        }

        int Font(float size)
        {
            return Mathf.Clamp(Mathf.RoundToInt(size * _zoom), 4, 40);
        }

        Vector2 Screen(Vector2 p)
        {
            return _offset + p * _zoom;
        }

        Rect Screen(Rect r)
        {
            return new Rect(_offset.x + r.x * _zoom, _offset.y + r.y * _zoom, r.width * _zoom, r.height * _zoom);
        }

        void Grid(Rect r)
        {
            float minor = 16f * _zoom;
            Lines(r, minor, MinorLine);
            Lines(r, minor * 8f, MajorLine);
        }

        void Lines(Rect r, float step, Color c)
        {
            if (step < 8f) return;
            for (float x = Mod(_offset.x, step); x < r.width; x += step) EditorGUI.DrawRect(new Rect(x, 0f, 1f, r.height), c);
            for (float y = Mod(_offset.y, step); y < r.height; y += step) EditorGUI.DrawRect(new Rect(0f, y, r.width, 1f), c);
        }

        static float Mod(float a, float b)
        {
            return a - b * Mathf.Floor(a / b);
        }

        Vector2 Out(Node n) => new Vector2(n.pos.x + NodeW, n.pos.y + PinY);
        Vector2 In(Node n) => new Vector2(n.pos.x, n.pos.y + PinY);
        Vector2 Top(Node n) => new Vector2(n.pos.x + NodeW * 0.5f, n.pos.y);
        Vector2 Bottom(Node n) => new Vector2(n.pos.x + NodeW * 0.5f, n.pos.y + NodeH);

        void Ends(Wire w, out Vector2 p0, out Vector2 c0, out Vector2 c1, out Vector2 p1)
        {
            Node a = _nodes[(int)w.a], b = _nodes[(int)w.b];
            switch (w.shape)
            {
                case Shape.Rise:
                {
                    p0 = Top(a);
                    p1 = In(b);
                    float dy = Mathf.Abs(p1.y - p0.y), dx = Mathf.Abs(p1.x - p0.x);
                    c0 = p0 + new Vector2(0f, -Mathf.Min(dy * 0.75f, 140f));
                    c1 = p1 - new Vector2(Mathf.Max(dx * 0.4f, 40f), 0f);
                    return;
                }
                case Shape.Up:
                    p0 = Top(a);
                    p1 = Bottom(b);
                    c0 = p0 + new Vector2(0f, -30f);
                    c1 = p1 + new Vector2(0f, 30f);
                    return;
                case Shape.Arch:
                case Shape.Loop:
                {
                    p0 = Top(a);
                    p1 = Top(b);
                    float top = Mathf.Min(p0.y, p1.y) - w.depth;
                    c0 = new Vector2(p0.x, top);
                    c1 = new Vector2(p1.x, top);
                    return;
                }
                case Shape.Sag:
                    p0 = Out(a);
                    p1 = In(b);
                    c0 = p0 + new Vector2(60f, w.depth);
                    c1 = p1 + new Vector2(-60f, w.depth);
                    return;
                default:
                {
                    p0 = Out(a);
                    p1 = In(b);
                    float d = Mathf.Max(Mathf.Abs(p1.x - p0.x) * 0.5f, 40f);
                    c0 = p0 + new Vector2(d, 0f);
                    c1 = p1 - new Vector2(d, 0f);
                    return;
                }
            }
        }

        static Vector2 Bez(Vector2 a, Vector2 ta, Vector2 tb, Vector2 b, float t)
        {
            float u = 1f - t;
            return u * u * u * a + 3f * u * u * t * ta + 3f * u * t * t * tb + t * t * t * b;
        }

        static float Level(List<Pulse> ps, double now)
        {
            float best = 0f;
            for (int i = ps.Count - 1; i >= 0; i--)
            {
                double t = now - ps[i].start;
                if (t > Hold + Fade)
                {
                    ps.RemoveAt(i);
                    continue;
                }
                if (t < 0.0) continue;
                float v = t < Attack ? (float)(t / Attack) : t < Hold ? 1f : 1f - (float)((t - Hold) / Fade);
                v = Mathf.Clamp01(v);
                v = v * v * (3f - 2f * v);
                best = Mathf.Max(best, v * ps[i].strength);
            }
            return best;
        }

        static Color Lighter(Color c, float k)
        {
            var o = Color.Lerp(c, Color.white, k);
            o.a = c.a;
            return o;
        }

        static Rect Grow(Rect r, float e)
        {
            return new Rect(r.x - e, r.y - e, r.width + 2f * e, r.height + 2f * e);
        }

        static void Fill(Rect r, Color c, float radius)
        {
            GUI.DrawTexture(r, Texture2D.whiteTexture, ScaleMode.StretchToFill, true, 0f, c, Vector4.zero, Vector4.one * radius);
        }

        Texture2D Grad
        {
            get
            {
                if (_grad != null) return _grad;
                _grad = new Texture2D(64, 16, TextureFormat.RGBA32, false)
                {
                    hideFlags = HideFlags.HideAndDontSave,
                    wrapMode = TextureWrapMode.Clamp,
                    filterMode = FilterMode.Bilinear,
                };
                for (int y = 0; y < 16; y++)
                    for (int x = 0; x < 64; x++)
                    {
                        float v = Mathf.Lerp(0.78f, 1f, y / 15f);
                        _grad.SetPixel(x, y, new Color(v, v, v, Mathf.Lerp(1f, 0.5f, x / 63f)));
                    }
                _grad.Apply();
                return _grad;
            }
        }

        void DrawWire(Wire w, double now)
        {
            Ends(w, out var p0, out var c0, out var c1, out var p1);
            Vector2 a = Screen(p0), b = Screen(p1), ta = Screen(c0), tb = Screen(c1);
            float lv = Level(w.pulses, now);
            var tint = Lighter(_nodes[(int)w.a].color, 0.25f);
            var idle = new Color(tint.r, tint.g, tint.b, 0.36f);
            float width = Mathf.Max(1.4f, 2.2f * _zoom);
            if (w.dashed) Dashed(a, ta, tb, b, idle, width);
            else Handles.DrawBezier(a, b, ta, tb, idle, null, width);
            if (lv <= 0.01f) return;
            var glow = Lighter(tint, 0.3f);
            glow.a = 0.24f * lv;
            Handles.DrawBezier(a, b, ta, tb, glow, null, width * 4.5f);
            var hot = Color.Lerp(tint, Color.white, 0.45f);
            hot.a = lv;
            Handles.DrawBezier(a, b, ta, tb, hot, null, width * 1.6f);
            float phase = (float)(now % 0.8) / 0.8f;
            Handles.color = new Color(1f, 1f, 1f, lv);
            float r = Mathf.Max(1.6f, 3.2f * _zoom);
            for (int k = 0; k < 3; k++)
                Handles.DrawSolidDisc(Bez(a, ta, tb, b, Mathf.Repeat(phase + k / 3f, 1f)), Vector3.forward, r);
        }

        static void Dashed(Vector2 a, Vector2 ta, Vector2 tb, Vector2 b, Color c, float width)
        {
            Handles.color = c;
            const int n = 48;
            var seg = new Vector3[2];
            for (int i = 0; i < n; i += 2)
            {
                seg[0] = Bez(a, ta, tb, b, i / (float)n);
                seg[1] = Bez(a, ta, tb, b, (i + 1) / (float)n);
                Handles.DrawAAPolyLine(width, seg);
            }
        }

        void DrawNode(Node n, double now)
        {
            if (n.pending != null && n.pendingAt <= now)
            {
                n.caption = n.pending;
                n.pending = null;
                n.pendingAt = double.MaxValue;
            }
            float lv = Level(n.pulses, now);
            var r = Screen(new Rect(n.pos.x, n.pos.y, NodeW, NodeH));
            float z = _zoom, rad = 6f * z;
            if (lv > 0.01f)
                for (int i = 3; i >= 1; i--)
                {
                    float e = (3f + 5f * i) * z;
                    var g = Lighter(n.color, 0.2f);
                    g.a = lv * (0.34f - 0.09f * i);
                    Fill(Grow(r, e), g, rad + e);
                }
            Fill(Grow(r, 1.5f * z), new Color(0f, 0f, 0f, 0.7f), rad + 1.5f * z);
            Fill(r, NodeBody, rad);
            var h = new Rect(r.x, r.y, r.width, HeaderH * z);
            var hc = Color.Lerp(n.color * 0.85f, Lighter(n.color, 0.35f), lv);
            hc.a = 1f;
            GUI.DrawTexture(h, Grad, ScaleMode.StretchToFill, true, 0f, hc, Vector4.zero, new Vector4(rad, rad, 0f, 0f));
            EditorGUI.DrawRect(new Rect(h.x, h.yMax - 1f, h.width, 1f), new Color(0f, 0f, 0f, 0.45f));
            if (lv > 0.01f)
            {
                var bc = Lighter(n.color, 0.5f);
                bc.a = lv;
                GUI.DrawTexture(Grow(r, 1.5f * z), Texture2D.whiteTexture, ScaleMode.StretchToFill, true, 0f, bc,
                    Vector4.one * Mathf.Max(1f, 2f * z), Vector4.one * (rad + 1.5f * z));
            }
            float pad = 9f * z;
            GUI.Label(new Rect(h.x + pad, h.y, h.width - 2f * pad, h.height), n.title, _title);
            _caption.normal.textColor = Color.Lerp(CaptionIdle, Color.white, lv);
            GUI.Label(new Rect(r.x + pad, h.yMax + (r.height - h.height - 20f * z) / 2f, r.width - 2f * pad, 20f * z),
                n.caption ?? "", _caption);
        }

        void DrawPins(Wire w, double now)
        {
            Ends(w, out var p0, out _, out _, out var p1);
            float lv = Level(w.pulses, now);
            var c = Lighter(_nodes[(int)w.a].color, 0.35f);
            Pin(Screen(p0), c, lv);
            Pin(Screen(p1), c, lv);
        }

        void Pin(Vector2 p, Color c, float lv)
        {
            float r = Mathf.Max(2.4f, 4.2f * _zoom);
            Handles.color = new Color(0.03f, 0.03f, 0.03f, 1f);
            Handles.DrawSolidDisc(p, Vector3.forward, r + Mathf.Max(0.8f, 1.2f * _zoom));
            c.a = 1f;
            Handles.color = c;
            Handles.DrawSolidDisc(p, Vector3.forward, r);
            Handles.color = Color.Lerp(new Color(0.06f, 0.06f, 0.06f, 1f), Color.white, lv);
            Handles.DrawSolidDisc(p, Vector3.forward, r * 0.5f);
        }
    }
}
