using System.Collections.Generic;
using UnityEngine;

namespace PredictiveSfx
{
    public class PredictiveSfxShowcase : MonoBehaviour
    {
        public AudioSource source;
        public AudioClip authoredFallback;
        public float interval = 0.5f;

        static readonly string[] Surfaces = { "grass", "dirt", "stone", "wood", "metal", "water" };

        int _surfaceIdx;
        bool _live = true;
        bool _autoWalk = true;
        bool _equipped = true;
        int _ammo = 6;

        float _t;
        int _steps, _shots, _reloads;
        readonly Dictionary<string, AudioClip> _frozen = new Dictionary<string, AudioClip>();
        readonly Queue<int> _recent = new Queue<int>();
        readonly HashSet<int> _unique = new HashSet<int>();
        int _identicalRun;
        int _lastVariant = -1;
        string _why = "press [Space] to A/B  -  LIVE varies every step, OLD WAY loops one clip";
        GUIStyle _style, _banner;

        string Surface => Surfaces[_surfaceIdx];

        void Start()
        {
            if (source == null) source = gameObject.AddComponent<AudioSource>();
            PushContext();
        }

        void PushContext()
        {
            PredictiveSfxService.Instance.SetContext(_equipped ? "gun" : null, new[] { Surface }, "field");
        }

        void Update()
        {
            HandleInput();
            if (_autoWalk)
            {
                _t += Time.deltaTime;
                if (_t >= interval) { _t = 0f; Step(); }
            }
        }

        void HandleInput()
        {
            if (Input.GetKeyDown(KeyCode.Space)) { _live = !_live; _why = _live ? "LIVE: a fresh generated variant every step - no two identical" : "OLD WAY: one authored clip per surface, looped - repetition fatigue"; }
            if (Input.GetKeyDown(KeyCode.M)) { _surfaceIdx = (_surfaceIdx + 1) % Surfaces.Length; PushContext(); _why = "same 'footstep' event, now on " + Surface + "  -  zero new audio files"; }
            if (Input.GetKeyDown(KeyCode.E)) { _equipped = !_equipped; PushContext(); _why = _equipped ? "equipped gun  -  shoot/reload/empty-click now exist purely from context" : "holstered the gun"; }
            if (Input.GetKeyDown(KeyCode.F)) Fire();
            if (Input.GetKeyDown(KeyCode.R)) Reload();
            if (Input.GetKeyDown(KeyCode.Tab)) _autoWalk = !_autoWalk;
            if (Input.GetKeyDown(KeyCode.V)) Step();
            if (Input.GetKeyDown(KeyCode.UpArrow)) interval = Mathf.Max(0.1f, interval - 0.05f);
            if (Input.GetKeyDown(KeyCode.DownArrow)) interval = Mathf.Min(1.5f, interval + 0.05f);
        }

        void Step()
        {
            _steps++;
            if (_live)
            {
                PredictiveSfxService.Instance.Request("footstep", Surface, "normal", "default", OnFootstep, OnFailed);
            }
            else if (_frozen.TryGetValue(Surface, out var clip) && clip != null)
            {
                source.PlayOneShot(clip);
            }
            else
            {
                PredictiveSfxService.Instance.Request("footstep", Surface, "normal", "default",
                    c => { _frozen[Surface] = c; source.PlayOneShot(c); }, OnFailed);
            }
        }

        void OnFootstep(AudioClip clip)
        {
            source.PlayOneShot(clip);
            int v = ParseVariant(clip.name);
            _unique.Add(v);
            _identicalRun = (v == _lastVariant) ? _identicalRun + 1 : 0;
            _lastVariant = v;
            _recent.Enqueue(v);
            while (_recent.Count > 8) _recent.Dequeue();
        }

        void Fire()
        {
            if (!_equipped) return;
            if (_ammo > 0)
            {
                _ammo--;
                _shots++;
                PredictiveSfxService.Instance.Request("shoot", "metal", "normal", "default", c => source.PlayOneShot(c), OnFailed);
            }
            else
            {
                PredictiveSfxService.Instance.Request("empty_click", "metal", "normal", "default", c => source.PlayOneShot(c), OnFailed);
            }
        }

        void Reload()
        {
            if (!_equipped) return;
            _reloads++;
            _ammo = 6;
            PredictiveSfxService.Instance.Request("reload", "metal", "normal", "default", c => source.PlayOneShot(c), OnFailed);
        }

        void OnFailed()
        {
            if (authoredFallback != null && source != null) source.PlayOneShot(authoredFallback);
        }

        static int ParseVariant(string name)
        {
            var parts = name.Split('|');
            if (parts.Length >= 4 && int.TryParse(parts[3], out var v)) return v;
            return -1;
        }

        void OnGUI()
        {
            if (_style == null)
            {
                _style = new GUIStyle(GUI.skin.label) { fontSize = 16, richText = true, wordWrap = true };
                _banner = new GUIStyle(GUI.skin.box) { fontSize = 18, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
            }
            var svc = PredictiveSfxService.Instance;
            float x = 10, y = 10, w = 400;
            var bg = GUI.backgroundColor;

            GUI.backgroundColor = _live ? new Color(0.12f, 0.6f, 0.22f) : new Color(0.72f, 0.16f, 0.16f);
            GUI.Box(new Rect(x, y, w, 32), _live ? "MODE: LIVE  (generated, varied)" : "MODE: OLD WAY  (one looped clip)", _banner);
            GUI.backgroundColor = bg;
            y += 40;

            GUI.Label(new Rect(x, y, w, 22), svc.IsConnected ? "<color=#5f5>sidecar: connected</color>" : "<color=#f66>sidecar: OFFLINE - run scripts/run_sidecar.py</color>", _style); y += 22;
            GUI.Label(new Rect(x, y, w, 22), "surface: " + Surface + "    equipped: " + (_equipped ? "gun (ammo " + _ammo + ")" : "none"), _style); y += 22;
            GUI.Label(new Rect(x, y, w, 22), "steps " + _steps + "   shots " + _shots + "   reloads " + _reloads + "   unique variants " + _unique.Count, _style); y += 22;
            GUI.Label(new Rect(x, y, w, 22), "served: " + (svc.LastServed ?? "-") + "    cached " + svc.CachedCount + " / fallback " + svc.FallbackCount, _style); y += 24;

            GUI.Label(new Rect(x, y, w, 22), "recent variants  (identical in a row: " + _identicalRun + ")", _style); y += 22;
            float cx = x;
            foreach (var v in _recent)
            {
                GUI.backgroundColor = VariantColor(v);
                GUI.Box(new Rect(cx, y, 30, 26), v.ToString());
                cx += 34;
            }
            GUI.backgroundColor = bg;
            y += 34;

            GUI.Label(new Rect(x, y, w, 46), "<b>why:</b> " + _why, _style); y += 48;
            GUI.Label(new Rect(x, y, w, 84), "[Space] OLD WAY / LIVE    [M] change surface    [E] equip gun\n[F] fire    [R] reload    [Tab] auto-walk    [V] single step\n[Up/Down] tempo", _style);
        }

        static Color VariantColor(int v)
        {
            if (v < 0) return Color.gray;
            return Color.HSVToRGB((v * 0.137f) % 1f, 0.6f, 0.92f);
        }
    }
}
