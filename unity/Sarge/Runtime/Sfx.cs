using System;
using System.Collections.Generic;
using UnityEngine;

namespace Sarge
{
    public static class Sfx
    {
        public static void Request(string eventType, string material, string intensity = "normal",
            Action<AudioClip> onReady = null, Action onFailed = null)
        {
            var svc = SargeService.Instance;
            svc.Request(eventType, material ?? "", intensity ?? "normal", svc.ContextOr(null), onReady, onFailed);
        }

        public static void Predict(string eventType, string material, string intensity, float etaSeconds,
            float confidence = 1f)
        {
            var svc = SargeService.Instance;
            svc.SendPredictions(new List<SfxPrediction>
            {
                new SfxPrediction
                {
                    source = "script",
                    eventType = eventType,
                    material = material ?? "",
                    intensity = intensity ?? "normal",
                    context = svc.ContextOr(null),
                    etaS = etaSeconds,
                    confidence = confidence,
                },
            }, Time.time);
        }
    }
}
