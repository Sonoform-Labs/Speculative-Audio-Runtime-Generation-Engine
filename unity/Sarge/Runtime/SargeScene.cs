using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.AI;

namespace Sarge
{
    public static class SargeScene
    {
        public const float ProbeInterval = 0.5f;
        public const float Hysteresis = 1f;
        public const float UpRay = 50f;
        public const float AxisRay = 30f;
        public const float ProjectileSpeed = 20f;
        public const float ProjectileAge = 3f;
        public const int MaxName = 48;

        static readonly Dictionary<string, string> _cache = new Dictionary<string, string>();
        static readonly Dictionary<int, bool> _projectile = new Dictionary<int, bool>();
        static AudioListener _listener;
        static AudioReverbZone[] _zones;
        static float _zonesAt = -100f;
        static string _room = "", _candidate = "";
        static float _candidateSince, _nextProbe;

        public static string Room => _room;
        public static int Probes { get; private set; }

        public static string ForEmitter(Component c, string sourceOverride = null)
        {
            if (c == null) return null;
            var a = c.GetComponent<AudioSource>();
            string act = a != null && a.loop ? "loop" : "use";
            string key = c.GetInstanceID().ToString(CultureInfo.InvariantCulture) + "|" + act + "|" + sourceOverride;
            if (!_cache.TryGetValue(key, out var head))
            {
                string src = string.IsNullOrEmpty(sourceOverride) ? SourceOf(c) : sourceOverride;
                head = Head(src, Root(c), Clean(c.gameObject.name), act);
                _cache[key] = head;
            }
            return WithRoom(head);
        }

        public static string ForActor(Component c, string act, string sourceOverride = null)
        {
            if (c == null) return null;
            string key = c.GetInstanceID().ToString(CultureInfo.InvariantCulture) + "|actor|" + act + "|" + sourceOverride;
            if (!_cache.TryGetValue(key, out var head))
            {
                string src = string.IsNullOrEmpty(sourceOverride) ? SourceOf(c) : sourceOverride;
                head = Head(src, Root(c), null, act);
                _cache[key] = head;
            }
            return WithRoom(head);
        }

        public static string ForPhysics(Component c, Rigidbody rb, float bornTime)
        {
            if (c == null) return null;
            string src = IsProjectile(rb, bornTime) ? "projectile" : "physics";
            return WithRoom(Head(src, Root(c), Clean(c.gameObject.name), null));
        }

        public static string SourceOf(Component c)
        {
            if (c == null) return "world";
            if (c.GetComponentInParent<Canvas>() != null) return "ui";
            var root = c.transform.root;
            if (IsPlayer(root)) return "player";
            if (root.GetComponentInChildren<NavMeshAgent>() != null || root.GetComponent<Animator>() != null) return "npc";
            return "world";
        }

        public static bool IsPlayer(Transform root)
        {
            if (root == null) return false;
            if (_listener == null) _listener = Object.FindObjectOfType<AudioListener>();
            if (_listener != null && _listener.transform.root == root) return true;
            try { return root.CompareTag("Player"); }
            catch (UnityException) { return false; }
        }

        public static bool IsProjectile(Rigidbody rb, float bornTime)
        {
            if (rb == null || rb.isKinematic) return false;
            int id = rb.GetInstanceID();
            if (_projectile.TryGetValue(id, out bool p)) return p;
            p = rb.collisionDetectionMode != CollisionDetectionMode.Discrete ||
                (Time.time - bornTime < ProjectileAge && rb.velocity.magnitude > ProjectileSpeed);
            _projectile[id] = p;
            return p;
        }

        public static void Tick(float now)
        {
            if (now < _nextProbe) return;
            _nextProbe = now + ProbeInterval;
            if (_listener == null) _listener = Object.FindObjectOfType<AudioListener>();
            if (_listener == null) return;
            string c = ProbeRoom(_listener.transform.position, now);
            Probes++;
            if (c != _candidate) { _candidate = c; _candidateSince = now; }
            if (c != _room && (string.IsNullOrEmpty(_room) || now - _candidateSince >= Hysteresis)) _room = c;
        }

        public static string ProbeRoom(Vector3 p, float now)
        {
            if (_zones == null || now - _zonesAt > 10f)
            {
                _zones = Object.FindObjectsOfType<AudioReverbZone>();
                _zonesAt = now;
            }
            foreach (var z in _zones)
            {
                if (z == null || !z.isActiveAndEnabled) continue;
                if (Vector3.Distance(z.transform.position, p) > z.minDistance) continue;
                string r = RoomOfPreset(z.reverbPreset);
                if (r != null) return r;
            }
            if (!Physics.Raycast(p, Vector3.up, UpRay, ~0, QueryTriggerInteraction.Ignore)) return "open";
            float wx = Ray(p, Vector3.right) + Ray(p, Vector3.left);
            float h = Ray(p, Vector3.up) + Ray(p, Vector3.down);
            float wz = Ray(p, Vector3.forward) + Ray(p, Vector3.back);
            float lo = Mathf.Min(wx, wz), hi = Mathf.Max(wx, wz);
            if (lo < 4f && hi > 4f * lo) return "corridor";
            float vol = wx * h * wz;
            return vol < 60f ? "small" : vol < 1000f ? "medium" : "large";
        }

        static float Ray(Vector3 p, Vector3 d)
        {
            return Physics.Raycast(p, d, out RaycastHit hit, AxisRay, ~0, QueryTriggerInteraction.Ignore) ? hit.distance : AxisRay;
        }

        public static string RoomOfPreset(AudioReverbPreset preset)
        {
            switch (preset)
            {
                case AudioReverbPreset.PaddedCell:
                case AudioReverbPreset.Room:
                case AudioReverbPreset.Bathroom:
                case AudioReverbPreset.Livingroom:
                case AudioReverbPreset.Stoneroom:
                    return "small";
                case AudioReverbPreset.Generic:
                    return "medium";
                case AudioReverbPreset.Auditorium:
                case AudioReverbPreset.Concerthall:
                case AudioReverbPreset.Cave:
                case AudioReverbPreset.Arena:
                case AudioReverbPreset.Hangar:
                    return "large";
                case AudioReverbPreset.CarpetedHallway:
                case AudioReverbPreset.Hallway:
                case AudioReverbPreset.StoneCorridor:
                case AudioReverbPreset.Alley:
                case AudioReverbPreset.SewerPipe:
                    return "corridor";
                case AudioReverbPreset.Forest:
                case AudioReverbPreset.City:
                case AudioReverbPreset.Mountains:
                case AudioReverbPreset.Quarry:
                case AudioReverbPreset.Plain:
                case AudioReverbPreset.ParkingLot:
                    return "open";
                default:
                    return null;
            }
        }

        public static void Clear()
        {
            _cache.Clear();
            _projectile.Clear();
            _listener = null;
            _zones = null;
        }

        public static void ResetRoom()
        {
            _room = "";
            _candidate = "";
            _nextProbe = 0f;
        }

        static string Root(Component c)
        {
            return Clean(c.transform.root.name);
        }

        static string Head(string src, string who, string obj, string act)
        {
            var sb = new StringBuilder(160);
            sb.Append("{\"src\":").Append(Quote(src ?? "world"));
            if (!string.IsNullOrEmpty(who)) sb.Append(",\"who\":").Append(Quote(who));
            if (!string.IsNullOrEmpty(obj)) sb.Append(",\"obj\":").Append(Quote(obj));
            if (!string.IsNullOrEmpty(act)) sb.Append(",\"act\":").Append(Quote(act));
            return sb.ToString();
        }

        static string WithRoom(string head)
        {
            var svc = SargeService.Instance;
            if (svc != null && svc.roomProbe && !string.IsNullOrEmpty(_room))
                return head + ",\"room\":" + Quote(_room) + "}";
            return head + "}";
        }

        public static string Clean(string s)
        {
            s = (s ?? "").Replace(" (Instance)", "").Replace("(Clone)", "").Trim();
            return s.Length > MaxName ? s.Substring(0, MaxName) : s;
        }

        static string Quote(string s)
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
            sb.Append('"');
            return sb.ToString();
        }
    }
}
