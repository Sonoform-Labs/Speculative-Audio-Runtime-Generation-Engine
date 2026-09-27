using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Sarge
{
    public class SargeImpulseProbe : MonoBehaviour
    {
        public float mass = 70f;
        public float fallThreshold = 0.6f;
        public float gaitInterval = 0.25f;
        public float movingThreshold = 0.2f;
        public int keepOnScreen = 10;
        public bool writeCsv = true;
        public bool logToConsole = true;

        CharacterController _cc;
        Rigidbody _rb;
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
        readonly List<string> _rows = new List<string>();
        readonly Queue<string> _screen = new Queue<string>();

        void Awake()
        {
            _cc = GetComponent<CharacterController>();
            _rb = GetComponent<Rigidbody>();
            _prevPos = transform.position;
            _csvPath = Path.Combine(Application.persistentDataPath,
                                    "sarge_impulse_probe.csv");
            _rows.Add("event,t,airborne_s,vy_before,vy_after,dv," +
                      "impulse_proxy_Ns,horizontal_speed");
        }

        string ControllerKind()
        {
            if (_cc != null) return "CharacterController";
            if (_rb != null) return _rb.isKinematic ? "kinematic Rigidbody" : "dynamic Rigidbody";
            return "custom transform controller (no physics component)";
        }

        void Update()
        {
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

        void OnCollisionEnter(Collision c)
        {
            _sawCollisionImpulse = true;
            _maxCollisionImpulse = Mathf.Max(_maxCollisionImpulse, c.impulse.magnitude);
        }

        void OnGUI()
        {
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
            sb.AppendLine();
            foreach (var s in _screen) sb.AppendLine(s);
            GUI.Label(new Rect(10, 10, 900, 640), sb.ToString());
        }

        void OnApplicationQuit()
        {
            if (writeCsv) Flush();
        }
    }
}
