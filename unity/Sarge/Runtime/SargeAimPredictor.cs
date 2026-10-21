using System;
using System.Collections.Generic;
using UnityEngine;

namespace Sarge
{
    public class SargeAimPredictor : MonoBehaviour
    {
        public Camera cam;
        public bool armed = false;
        public float interval = 0.1f;
        public float holdSeconds = 0.15f;
        public float maxDistance = 100f;
        public string eventType = "";
        public string after = "";
        public string intensity = "normal";
        public string context = "";
        public string defaultMaterial = "";
        public float etaSeconds = 0.5f;
        public float confidence = 0.6f;
        public LayerMask mask = ~0;

        public static Func<Collider, string> MaterialResolver;
        public static Func<Collider, string> EventResolver;
        public static Func<RaycastHit, string> PhysResolver;
        public static Func<RaycastHit, string> SceneResolver;

        string _candidate;
        float _since;
        string _sent, _sentId;
        float _next;

        public int Sent { get; private set; }
        public string LastMaterial => _sent;

        void Update()
        {
            if (!armed) { _sent = null; _sentId = null; _candidate = null; return; }
            if (Time.time < _next) return;
            _next = Time.time + interval;
            var c = cam != null ? cam : Camera.main;
            if (c == null) return;
            if (!Physics.Raycast(c.transform.position, c.transform.forward, out RaycastHit hit, maxDistance, mask,
                    QueryTriggerInteraction.Ignore))
            {
                _candidate = null;
                return;
            }
            string mat = (MaterialResolver != null ? MaterialResolver(hit.collider) : null)
                         ?? SargeAutoCapture.MaterialOf(hit.collider) ?? defaultMaterial ?? "";
            if (mat != _candidate)
            {
                _candidate = mat;
                _since = Time.time;
                return;
            }
            if (Time.time - _since < holdSeconds) return;
            string resolved = EventResolver != null ? EventResolver(hit.collider) : null;
            string phys = PhysResolver != null ? PhysResolver(hit) : null;
            string sig = string.IsNullOrEmpty(phys) ? null : SargePhysics.Signature(hit.collider, null);
            string id = mat + "|" + resolved + "|" + sig;
            if (id == _sentId) return;
            _sentId = id;
            _sent = mat;
            var p = new SfxPrediction
            {
                source = "aim",
                eventType = resolved ?? eventType ?? "",
                material = mat,
                intensity = intensity,
                context = SargeService.Instance.ContextOr(context),
                etaS = etaSeconds,
                confidence = confidence,
                after = resolved != null || string.IsNullOrEmpty(after) ? null : after,
                physJson = phys,
                physSig = sig,
                scene = SceneResolver != null && SargeService.Instance.sendScene ? SceneResolver(hit) : null,
            };
            SargeService.Instance.SendPredictions(new List<SfxPrediction>(1) { p }, Time.time);
            Sent++;
        }
    }
}
