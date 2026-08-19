using UnityEngine;

namespace Sage
{
    [RequireComponent(typeof(AudioSource))]
    public class SageAutoCapture : MonoBehaviour
    {
        public AudioSource source;
        public string eventType = "impact";
        public string defaultMaterial = "stone";
        public float minVelocity = 0.6f;
        public float softVelocity = 1.5f;
        public float hardVelocity = 6f;
        public float cooldown = 0.05f;

        float _last;

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        void OnCollisionEnter(Collision c)
        {
            float v = c.relativeVelocity.magnitude;
            if (v < minVelocity) return;
            if (Time.time - _last < cooldown) return;
            _last = Time.time;

            string mat = MaterialOf(c.collider) ?? defaultMaterial;
            string intensity = v >= hardVelocity ? "hard" : (v <= softVelocity ? "soft" : "normal");
            var src = source;
            SageService.Instance.Request(eventType, mat, intensity, "default",
                clip => { if (src != null) src.PlayOneShot(clip); }, null);
        }

        static string MaterialOf(Collider col)
        {
            if (col == null) return null;
            var surf = col.GetComponentInParent<SageSurface>();
            if (surf != null && !string.IsNullOrEmpty(surf.material)) return surf.material;
            var pm = col.sharedMaterial;
            if (pm != null && !string.IsNullOrEmpty(pm.name))
                return pm.name.Replace(" (Instance)", "").Trim().ToLower();
            return null;
        }
    }
}
