using System.Collections.Generic;
using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(Rigidbody))]
    public class SargeImpactPredictor : MonoBehaviour
    {
        public SargeAutoCapture capture;
        public float interval = 0.1f;
        public float step = 0.02f;
        public float horizon = 2.5f;
        public float minEta = 0.02f;
        public float kickThreshold = 1f;
        public float confidence = 0.9f;
        public float minSendInterval = 0.05f;
        public LayerMask mask = ~0;
        public string context = "default";

        public static int castsPerFrame = 1000;
        static int _castFrame = -1, _casts;

        Rigidbody _rb;
        Collider _col;
        float _next, _lastSend = -10f;
        int _lastTarget;
        string _lastIntensity;
        Vector3 _lastVel;
        readonly RaycastHit[] _hits = new RaycastHit[8];
        readonly Dictionary<int, Vector3> _touch = new Dictionary<int, Vector3>();
        readonly HashSet<int> _support = new HashSet<int>();

        public int Sent { get; private set; }
        public float LastEta { get; private set; }

        void Awake()
        {
            _rb = GetComponent<Rigidbody>();
            _col = GetComponent<Collider>();
            if (capture == null) capture = GetComponent<SargeAutoCapture>();
        }

        void OnCollisionEnter(Collision c)
        {
            _lastTarget = 0;
            _lastIntensity = null;
            if (c.collider != null) _touch[c.collider.GetInstanceID()] = c.contactCount > 0 ? c.GetContact(0).normal : Vector3.up;
        }

        void OnCollisionExit(Collision c)
        {
            if (c.collider != null) _touch.Remove(c.collider.GetInstanceID());
        }

        void OnDisable()
        {
            _touch.Clear();
        }

        void FixedUpdate()
        {
            if (_rb == null) return;
            Vector3 v = _rb.velocity;
            bool kicked = (v - _lastVel).magnitude > kickThreshold;
            _lastVel = v;
            if (kicked) Sample();
        }

        void Update()
        {
            if (Time.time >= _next) Sample();
        }

        void Sample()
        {
            _next = Time.time + interval;
            if (capture == null || _rb == null || _col == null) return;
            if (_rb.isKinematic || _rb.IsSleeping()) return;
            Vector3 v = _rb.velocity;
            if (v.magnitude < capture.minVelocity) return;
            if (Time.frameCount != _castFrame) { _castFrame = Time.frameCount; _casts = 0; }
            if (_casts >= castsPerFrame) { _next = Time.time; return; }
            Vector3 g = _rb.useGravity ? Physics.gravity : Vector3.zero;
            _support.Clear();
            foreach (var kv in _touch)
            {
                var n = kv.Value;
                if (Mathf.Abs(Vector3.Dot(v, n)) >= capture.minVelocity) continue;
                _support.Add(kv.Key);
                g -= Vector3.Dot(g, n) * n;
            }
            Vector3 p = _col.bounds.center;
            float radius = Radius() * 0.95f;
            for (float t = 0f; t < horizon; t += step)
            {
                if (_casts++ >= castsPerFrame) { _next = Time.time; return; }
                Vector3 p1 = p + v * step + 0.5f * step * step * g;
                Vector3 d = p1 - p;
                float len = d.magnitude;
                if (len > 1e-5f && Cast(p, radius, d / len, len, out RaycastHit hit))
                {
                    float frac = hit.distance / len;
                    float eta = t + frac * step;
                    if (eta < minEta) return;
                    Predict(hit, v + g * (frac * step), eta);
                    return;
                }
                p = p1;
                v += g * step;
            }
        }

        float Radius()
        {
            if (_col is SphereCollider s)
            {
                var ls = transform.lossyScale;
                return s.radius * Mathf.Max(Mathf.Abs(ls.x), Mathf.Max(Mathf.Abs(ls.y), Mathf.Abs(ls.z)));
            }
            var e = _col.bounds.extents;
            return Mathf.Min(e.x, Mathf.Min(e.y, e.z));
        }

        bool Cast(Vector3 from, float radius, Vector3 dir, float len, out RaycastHit best)
        {
            best = default;
            int n = Physics.SphereCastNonAlloc(from, radius, dir, _hits, len, mask, QueryTriggerInteraction.Ignore);
            bool found = false;
            for (int i = 0; i < n; i++)
            {
                var h = _hits[i];
                if (h.collider == null || h.collider == _col || h.collider.attachedRigidbody == _rb) continue;
                if (_support.Contains(h.collider.GetInstanceID())) continue;
                if (h.distance <= 0f && h.point == Vector3.zero) continue;
                if (!found || h.distance < best.distance) { best = h; found = true; }
            }
            return found;
        }

        void Predict(RaycastHit hit, Vector3 vel, float eta)
        {
            var other = hit.rigidbody;
            Vector3 rel = vel - (other != null ? other.GetPointVelocity(hit.point) : Vector3.zero);
            float speed = rel.magnitude;
            if (speed < capture.minVelocity) return;
            string intensity = capture.IntensityFor(speed);
            int target = hit.collider.GetInstanceID();
            if (target == _lastTarget && intensity == _lastIntensity) return;
            if (Time.time - _lastSend < minSendInterval) return;
            _lastSend = Time.time;
            _lastTarget = target;
            _lastIntensity = intensity;
            string mat = SargeAutoCapture.MaterialOf(hit.collider) ?? capture.defaultMaterial;
            string phys = null;
            if (capture.sendPhysics)
            {
                float vn = Mathf.Abs(Vector3.Dot(rel, hit.normal));
                float e = _col.sharedMaterial != null ? _col.sharedMaterial.bounciness : 0f;
                phys = SargePhysics.Json(_rb, _col, hit.collider, rel, hit.normal, _rb.mass * vn * (1f + e),
                    SargePhysics.ListenerDistance(hit.point));
            }
            var preds = new List<SfxPrediction>(1)
            {
                new SfxPrediction
                {
                    source = "impact",
                    eventType = capture.eventType,
                    material = mat,
                    intensity = intensity,
                    context = context,
                    etaS = eta,
                    confidence = confidence,
                    physJson = phys,
                    physSig = phys != null ? SargePhysics.Signature(_col, hit.collider) : null,
                },
            };
            SargeService.Instance.SendPredictions(preds, Time.time);
            Sent++;
            LastEta = eta;
        }
    }
}
