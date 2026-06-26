using System.Collections.Generic;
using UnityEngine;

namespace PredictiveSfx
{
    public class PredictiveSfxMotionPredictor : MonoBehaviour
    {
        public float strideMeters = 0.75f;
        public float leadSeconds = 0.6f;
        public float minSpeed = 0.4f;
        public float rayDistance = 2f;
        public float stabilityAlpha = 0.4f;
        public string defaultMaterial = "grass";
        public string intensity = "normal";

        Vector3 _lastPos;
        Vector3 _lastVel;
        bool _have;
        float _dist;
        float _nextStepDist;
        int _stepCount;
        int _emittedStep = -1;
        float _stab = 1f;

        void Start()
        {
            _lastPos = transform.position;
            _nextStepDist = strideMeters;
        }

        void Update()
        {
            float dt = Time.deltaTime;
            if (dt <= 0f) return;
            Vector3 pos = transform.position;
            if (!_have) { _lastPos = pos; _have = true; return; }

            Vector3 d = pos - _lastPos;
            d.y = 0f;
            float dist = d.magnitude;
            float speed = dist / dt;
            Vector3 vel = d / dt;

            float stable = 1f;
            if (_lastVel.sqrMagnitude > 1e-6f && vel.sqrMagnitude > 1e-6f)
                stable = Mathf.Max(0f, Vector3.Dot(vel.normalized, _lastVel.normalized));
            _stab = (1f - stabilityAlpha) * _stab + stabilityAlpha * stable;

            _dist += dist;
            _lastVel = vel;
            _lastPos = pos;

            if (speed < minSpeed) return;

            while (_nextStepDist - _dist <= 0f) { _nextStepDist += strideMeters; _stepCount++; }
            float remaining = _nextStepDist - _dist;
            float ttn = remaining / speed;
            if (ttn > leadSeconds || _emittedStep >= _stepCount) return;

            Vector3 footPos = pos + vel * ttn;
            string surface = SurfaceAt(footPos);
            var preds = new List<SfxPrediction>(1)
            {
                new SfxPrediction
                {
                    source = "motion",
                    eventType = "footstep",
                    material = surface,
                    intensity = intensity,
                    context = "default",
                    etaS = ttn,
                    confidence = _stab,
                },
            };
            PredictiveSfxService.Instance.SendPredictions(preds, Time.time);
            _emittedStep = _stepCount;
        }

        string SurfaceAt(Vector3 worldPos)
        {
            RaycastHit hit;
            if (Physics.Raycast(worldPos + Vector3.up * 0.5f, Vector3.down, out hit, rayDistance + 0.5f))
            {
                var surf = hit.collider.GetComponentInParent<PredictiveSfxSurface>();
                if (surf != null && !string.IsNullOrEmpty(surf.material)) return surf.material;
            }
            return defaultMaterial;
        }
    }
}
