using System.Collections.Generic;
using UnityEngine;

namespace Sarge
{
    public class SargeMotionPredictor : MonoBehaviour
    {
        public float strideMeters = 0.75f;
        public float leadSeconds = 0.6f;
        public float minSpeed = 0.4f;
        public float rayDistance = 2f;
        public float stabilityAlpha = 0.4f;
        public string eventType = "";
        public string defaultMaterial = "";
        public string intensity = "normal";
        public string context = "";

        public bool showMarkers = true;
        public float markerSize = 0.28f;
        public float markerLifetime = 0.9f;

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

            float horizon = Mathf.Max(leadSeconds, ttn);
            Vector3 footPos = pos + vel * horizon;
            Vector3 hitPoint;
            string surface = SurfaceAt(footPos, out hitPoint);
            Vector3 hereHit;
            string here = SurfaceAt(pos, out hereHit);
            SpawnMarker(hitPoint, _stab, surface != here);
            var preds = new List<SfxPrediction>(1)
            {
                new SfxPrediction
                {
                    source = "motion",
                    eventType = eventType ?? "",
                    material = surface ?? "",
                    intensity = intensity,
                    context = SargeService.Instance.ContextOr(context),
                    etaS = horizon,
                    confidence = _stab,
                    scene = StepScene(),
                },
            };
            SargeService.Instance.SendPredictions(preds, Time.time);
            _emittedStep = _stepCount;
        }

        string StepScene()
        {
            if (!SargeService.Instance.sendScene) return null;
            var steps = transform.root.GetComponentInChildren<SargeFootsteps>();
            return steps != null ? SargeScene.ForActor(steps, "step") : null;
        }

        string SurfaceAt(Vector3 worldPos, out Vector3 hitPoint)
        {
            hitPoint = worldPos;
            RaycastHit hit;
            if (Physics.Raycast(worldPos + Vector3.up * 0.5f, Vector3.down, out hit, rayDistance + 0.5f))
            {
                hitPoint = hit.point;
                var surf = hit.collider.GetComponentInParent<SargeSurface>();
                if (surf != null && !string.IsNullOrEmpty(surf.material)) return surf.material;
            }
            return defaultMaterial;
        }

        void SpawnMarker(Vector3 pos, float confidence, bool transition)
        {
            if (!showMarkers) return;
            var m = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            var col = m.GetComponent<Collider>();
            if (col != null) Destroy(col);
            m.layer = 2;
            m.name = "psfx_prediction_marker";
            m.transform.position = pos + Vector3.up * 0.04f;
            float s = transition ? markerSize * 1.9f : markerSize;
            m.transform.localScale = new Vector3(s, s * 0.22f, s);
            Color c = transition
                ? new Color(1f, 0.82f, 0.1f)
                : Color.Lerp(new Color(1f, 0.3f, 0.2f), new Color(0.2f, 1f, 0.35f), confidence);
            var mr = m.GetComponent<Renderer>();
            mr.material.color = c;
            if (mr.material.HasProperty("_BaseColor")) mr.material.SetColor("_BaseColor", c);
            if (mr.material.HasProperty("_EmissionColor"))
            {
                mr.material.EnableKeyword("_EMISSION");
                mr.material.SetColor("_EmissionColor", c * (transition ? 1.2f : 0.5f));
            }
            Destroy(m, transition ? markerLifetime * 1.7f : markerLifetime);
        }
    }
}
