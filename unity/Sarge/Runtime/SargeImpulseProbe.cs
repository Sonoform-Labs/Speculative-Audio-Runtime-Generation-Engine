using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Sarge
{
    public class SargeImpulseProbe : MonoBehaviour
    {
        public enum Mode { Auto, Motion, Collisions }

        public Mode mode = Mode.Auto;
        public float mass = 70f;
        public float fallThreshold = 0.6f;
        public float gaitInterval = 0.25f;
        public float movingThreshold = 0.2f;
        public int keepOnScreen = 10;
        public bool writeCsv = true;
        public bool writePhys = false;
        public bool logToConsole = true;
        public KeyCode kickKey = KeyCode.K;
        public float kickMin = 3f;
        public float kickMax = 14f;

        const string HitHeader = "object,t,impulse_Ns,rel_speed,other";
#if ENABLE_LEGACY_INPUT_MANAGER
        static readonly bool CanKick = true;
#else
        static readonly bool CanKick = false;
#endif

        static readonly List<SargeImpulseProbe> _bouncers = new List<SargeImpulseProbe>();
        static readonly HashSet<string> _labels = new HashSet<string>();

        CharacterController _cc;
        Rigidbody _rb;
        bool _motion;
        bool _collide;
        Vector3 _prevPos;
        bool _falling;
        float _minVy;
        float _fallStart;
        float _nextGait;
        int _landings;
        int _gait;
        float _minSpeed = float.MaxValue;
        float _maxSpeed;
        bool _sawCollisionImpulse;
        float _maxCollisionImpulse;
        string _csvPath;
        string _label;
        string _hitPath;
        bool _hitStarted;
        bool _hitFailed;
        int _hits;
        float _minHit = float.MaxValue;
        float _maxHit;
        float _lastHit;
        readonly List<string> _rows = new List<string>();
        readonly Queue<string> _screen = new Queue<string>();

        void Awake()
        {
            _cc = GetComponent<CharacterController>();
            _rb = GetComponent<Rigidbody>();
            bool dynamicBody = _cc == null && _rb != null && !_rb.isKinematic;
            _motion = mode == Mode.Motion || (mode == Mode.Auto && !dynamicBody);
            _collide = mode != Mode.Motion;
            _prevPos = transform.position;
            _csvPath = Path.Combine(Application.persistentDataPath,
                                    "sarge_impulse_probe.csv");
            if (_motion)
                _rows.Add("event,t,airborne_s,vy_before,vy_after,dv," +
                          "impulse_proxy_Ns,horizontal_speed");
            _label = Claim(Clean(name));
            _hitPath = Path.Combine(Application.persistentDataPath,
                                    "sarge_impulse_probe_" + _label + ".csv");
            if (_collide && !_motion && writeCsv) StartHitCsv();
        }

        void OnEnable()
        {
            if (!_motion && !_bouncers.Contains(this)) _bouncers.Add(this);
        }

        void OnDisable()
        {
            _bouncers.Remove(this);
        }

        void OnDestroy()
        {
            _labels.Remove(_label);
        }

        static string Clean(string s)
        {
            var sb = new StringBuilder();
            foreach (char ch in s ?? "")
                sb.Append(char.IsLetterOrDigit(ch) || ch == '_' || ch == '-' || ch == '.' ? ch : '_');
            return sb.Length > 0 ? sb.ToString() : "object";
        }

        static string Claim(string label)
        {
            string l = label;
            for (int i = 2; _labels.Contains(l); i++) l = label + "_" + i;
            _labels.Add(l);
            return l;
        }

        string ControllerKind()
        {
            if (_cc != null) return "CharacterController";
            if (_rb != null) return _rb.isKinematic ? "kinematic Rigidbody" : "dynamic Rigidbody";
            return "custom transform controller (no physics component)";
        }

        void Update()
        {
            if (!_motion)
            {
#if ENABLE_LEGACY_INPUT_MANAGER
                if (Input.GetKeyDown(kickKey)) Kick();
#endif
                return;
            }

            float dt = Time.deltaTime;
            if (dt <= 0f) return;

            Vector3 pos = transform.position;
            Vector3 vel = (pos - _prevPos) / dt;
            _prevPos = pos;
            float horiz = new Vector2(vel.x, vel.z).magnitude;

            if (!_falling && vel.y < -fallThreshold)
            {
                _falling = true;
                _minVy = vel.y;
                _fallStart = Time.time;
            }

            if (_falling)
            {
                if (vel.y < _minVy) _minVy = vel.y;
                if (vel.y > -fallThreshold * 0.25f)
                {
                    RecordLanding(_minVy, vel.y, Time.time - _fallStart, horiz);
                    _falling = false;
                }
            }
            else if (horiz > movingThreshold && Time.time >= _nextGait)
            {
                _nextGait = Time.time + gaitInterval;
                RecordGait(horiz);
            }
        }

        public void Kick()
        {
            if (_rb == null || _rb.isKinematic) return;
            _rb.velocity = Vector3.up * Random.Range(kickMin, kickMax);
            _rb.WakeUp();
        }

        void RecordLanding(float vyBefore, float vyAfter, float air, float horiz)
        {
            _landings++;
            float dv = Mathf.Abs(vyAfter - vyBefore);
            float impulse = mass * dv;
            _rows.Add(string.Format(CultureInfo.InvariantCulture,
                "landing,{0:F3},{1:F3},{2:F3},{3:F3},{4:F3},{5:F1},{6:F3}",
                Time.time, air, vyBefore, vyAfter, dv, impulse, horiz));
            Push(string.Format(CultureInfo.InvariantCulture,
                "landing {0}  air {1:F2}s  vy {2:F2}  dv {3:F2}  impulse {4:F0} Ns",
                _landings, air, vyBefore, dv, impulse));
        }

        void RecordGait(float horiz)
        {
            _gait++;
            _minSpeed = Mathf.Min(_minSpeed, horiz);
            _maxSpeed = Mathf.Max(_maxSpeed, horiz);
            _rows.Add(string.Format(CultureInfo.InvariantCulture,
                "gait,{0:F3},0,0,0,0,0,{1:F3}", Time.time, horiz));
        }

        void Push(string line)
        {
            _screen.Enqueue(line);
            while (_screen.Count > keepOnScreen) _screen.Dequeue();
            if (logToConsole) Debug.Log("[SargeImpulseProbe] " + line);
            if (writeCsv) Flush();
        }

        void Flush()
        {
            try
            {
                File.WriteAllText(_csvPath, string.Join("\n", _rows.ToArray()));
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("[SargeImpulseProbe] csv write failed: " + e.Message);
            }
        }

        void StartHitCsv()
        {
            _hitStarted = true;
            try
            {
                File.WriteAllText(_hitPath, HitHeader + (writePhys ? ",phys" : "") + "\n");
            }
            catch (System.Exception e)
            {
                _hitFailed = true;
                Debug.LogWarning("[SargeImpulseProbe] csv write failed: " + e.Message);
            }
        }

        void AppendHit(string row)
        {
            if (!_hitStarted) StartHitCsv();
            if (_hitFailed) return;
            try
            {
                File.AppendAllText(_hitPath, row + "\n");
            }
            catch (System.Exception e)
            {
                _hitFailed = true;
                Debug.LogWarning("[SargeImpulseProbe] csv write failed: " + e.Message);
            }
        }

        void OnCollisionEnter(Collision c)
        {
            float impulse = c.impulse.magnitude;
            _sawCollisionImpulse = true;
            _maxCollisionImpulse = Mathf.Max(_maxCollisionImpulse, impulse);
            if (!_collide) return;

            _hits++;
            _lastHit = impulse;
            _minHit = Mathf.Min(_minHit, impulse);
            _maxHit = Mathf.Max(_maxHit, impulse);
            string other = c.collider != null ? Clean(c.collider.name) : "none";
            string row = string.Format(CultureInfo.InvariantCulture,
                "{0},{1:F4},{2:F5},{3:F5},{4}",
                _label, Time.time, impulse, c.relativeVelocity.magnitude, other);
            if (writePhys)
            {
                Collider self = c.contactCount > 0 ? c.GetContact(0).thisCollider : GetComponent<Collider>();
                row += ",\"" + (SargePhysics.FromCollision(c, self) ?? "").Replace("\"", "\"\"") + "\"";
            }
            if (writeCsv) AppendHit(row);
            if (logToConsole) Debug.Log("[SargeImpulseProbe] " + row);
        }

        string HitSummary()
        {
            if (_hits == 0) return _label + ": no collisions yet";
            return string.Format(CultureInfo.InvariantCulture,
                "{0}: {1} hits   impulse {2:F2} to {3:F2} Ns   last {4:F2}",
                _label, _hits, _minHit, _maxHit, _lastHit);
        }

        void OnGUI()
        {
            if (!_motion)
            {
                int i = _bouncers.IndexOf(this);
                if (i < 0) return;
                float x = Mathf.Max(10f, Screen.width - 570f);
                float y = Screen.height - 10f - 22f * (_bouncers.Count + 1);
                if (i == 0)
                    GUI.Label(new Rect(x, y, 560, 22),
                              "SARGE impulse probe, collisions" +
                              (CanKick ? "   press " + kickKey + " to kick the bouncers" : ""));
                GUI.Label(new Rect(x, y + 22f * (i + 1), 560, 22), HitSummary());
                return;
            }

            var sb = new StringBuilder();
            sb.AppendLine("SARGE impulse probe");
            sb.AppendLine();
            sb.AppendLine("controller: " + ControllerKind());
            sb.AppendLine("native Collision.impulse received: " +
                          (_sawCollisionImpulse
                           ? _maxCollisionImpulse.ToString("F1") + " Ns max"
                           : "no"));
            sb.AppendLine();
            sb.AppendLine("landings (jumps and drops): " + _landings +
                          "   impulse = mass * |delta vy|");
            sb.AppendLine("gait samples while walking: " + _gait +
                          (_gait > 0
                           ? string.Format(CultureInfo.InvariantCulture,
                               "   speed {0:F2} to {1:F2} m/s", _minSpeed, _maxSpeed)
                           : ""));
            sb.AppendLine();
            sb.AppendLine("Walking on flat ground has no vertical velocity change, so a");
            sb.AppendLine("footfall carries no derivable impulse here. Only jumps and drops do.");
            sb.AppendLine("Footstep intensity has to come from gait: speed and cadence.");
            sb.AppendLine();
            if (writeCsv) sb.AppendLine("csv: " + _csvPath);
            if (_hits > 0) sb.AppendLine(HitSummary() + "   csv: " + _hitPath);
            sb.AppendLine();
            foreach (var s in _screen) sb.AppendLine(s);
            GUI.Label(new Rect(10, 10, 900, 640), sb.ToString());
        }

        void OnApplicationQuit()
        {
            if (writeCsv && _motion) Flush();
        }
    }
}
