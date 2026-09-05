using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(CharacterController))]
    public class SargeImpulseProbe : MonoBehaviour
    {
        public float mass = 70f;
        public int keepOnScreen = 10;
        public bool writeCsv = true;
        public bool logToConsole = true;

        CharacterController _cc;
        Rigidbody _rb;
        bool _wasGrounded;
        float _prevVy;
        Vector3 _prevVel;
        float _leftGroundAt;
        float _lastNormalY = 1f;
        bool _sawHit;
        bool _sawRigidbodyImpulse;
        float _maxRigidbodyImpulse;
        int _landings;
        string _csvPath;
        readonly List<string> _rows = new List<string>();
        readonly Queue<string> _screen = new Queue<string>();

        void Awake()
        {
            _cc = GetComponent<CharacterController>();
            _rb = GetComponent<Rigidbody>();
            _csvPath = Path.Combine(Application.persistentDataPath,
                                    "sarge_impulse_probe.csv");
            _rows.Add("landing,t,airborne_s,vy_before,vy_after,dv," +
                      "impulse_proxy_Ns,normal_y,horizontal_speed");
            _wasGrounded = _cc.isGrounded;
        }

        void Update()
        {
            bool grounded = _cc.isGrounded;
            Vector3 vel = _cc.velocity;

            if (!grounded && _wasGrounded)
                _leftGroundAt = Time.time;

            if (grounded && !_wasGrounded)
                Record(vel);

            _wasGrounded = grounded;
            _prevVy = vel.y;
            _prevVel = vel;
        }

        void Record(Vector3 velAfter)
        {
            _landings++;
            float vyBefore = _prevVy;
            float vyAfter = velAfter.y;
            float dv = Mathf.Abs(vyAfter - vyBefore);
            float impulse = mass * dv;
            float air = Mathf.Max(0f, Time.time - _leftGroundAt);
            float horiz = new Vector2(_prevVel.x, _prevVel.z).magnitude;

            _rows.Add(string.Format(CultureInfo.InvariantCulture,
                "{0},{1:F3},{2:F3},{3:F3},{4:F3},{5:F3},{6:F1},{7:F3},{8:F3}",
                _landings, Time.time, air, vyBefore, vyAfter, dv, impulse,
                _lastNormalY, horiz));

            string line = string.Format(CultureInfo.InvariantCulture,
                "landing {0}  air {1:F2}s  vy {2:F2}  dv {3:F2}  impulse {4:F0} Ns",
                _landings, air, vyBefore, dv, impulse);

            _screen.Enqueue(line);
            while (_screen.Count > keepOnScreen)
                _screen.Dequeue();

            if (logToConsole)
                Debug.Log("[SargeImpulseProbe] " + line);

            if (writeCsv)
                Flush();
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

        void OnControllerColliderHit(ControllerColliderHit hit)
        {
            _sawHit = true;
            _lastNormalY = hit.normal.y;
        }

        void OnCollisionEnter(Collision c)
        {
            _sawRigidbodyImpulse = true;
            _maxRigidbodyImpulse = Mathf.Max(_maxRigidbodyImpulse, c.impulse.magnitude);
        }

        void OnGUI()
        {
            var sb = new StringBuilder();
            sb.AppendLine("SARGE impulse probe");
            sb.AppendLine();
            sb.AppendLine("CharacterController is kinematic.");
            sb.AppendLine("ControllerColliderHit exposes: collider, controller, " +
                          "gameObject, moveDirection, moveLength, normal, point, " +
                          "rigidbody, transform.");
            sb.AppendLine("It exposes NO impulse and NO force field.");
            sb.AppendLine();
            sb.AppendLine("Rigidbody present on this object: " +
                          (_rb != null ? "yes" : "no"));
            sb.AppendLine("Collision.impulse ever received: " +
                          (_sawRigidbodyImpulse
                           ? _maxRigidbodyImpulse.ToString("F1") + " Ns max"
                           : "no"));
            sb.AppendLine("OnControllerColliderHit fired: " + (_sawHit ? "yes" : "no"));
            sb.AppendLine();
            sb.AppendLine("Derived impulse proxy = mass * |delta vy| at the " +
                          "airborne to grounded edge.");
            sb.AppendLine("landings: " + _landings);
            if (writeCsv)
                sb.AppendLine("csv: " + _csvPath);
            sb.AppendLine();
            foreach (var s in _screen)
                sb.AppendLine(s);

            GUI.Label(new Rect(10, 10, 900, 620), sb.ToString());
        }

        void OnApplicationQuit()
        {
            if (writeCsv)
                Flush();
        }
    }
}
