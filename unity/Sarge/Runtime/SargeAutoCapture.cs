using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(AudioSource))]
    public class SargeAutoCapture : MonoBehaviour
    {
        public AudioSource source;
        public string eventType = "impact";
        public string defaultMaterial = "stone";
        public float minVelocity = 0.6f;
        public float softVelocity = 1.5f;
        public float hardVelocity = 6f;
        public float cooldown = 0.05f;
        public bool sendPhysics = false;
        public bool speedGain = false;
        public float speedGainMaxDb = 6f;

        float _last;
        Collider _col;

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
            _col = GetComponent<Collider>();
        }

        void OnCollisionEnter(Collision c)
        {
            float v = c.relativeVelocity.magnitude;
            if (v < minVelocity) return;
            if (Time.time - _last < cooldown) return;
            _last = Time.time;

            string mat = MaterialOf(c.collider) ?? defaultMaterial;
            string intensity = IntensityFor(v);
            var src = source;
            if (!sendPhysics && !speedGain)
            {
                SargeService.Instance.Request(eventType, mat, intensity, "default",
                    clip => { if (src != null) src.PlayOneShot(clip); }, null);
                return;
            }
            float vol = speedGain ? SpeedGain(NormalSpeed(c)) : 1f;
            Collider self = c.contactCount > 0 ? c.GetContact(0).thisCollider : _col;
            string phys = sendPhysics ? SargePhysics.FromCollision(c, self) : null;
            string sig = sendPhysics ? SargePhysics.Signature(self, c.collider) : null;
            SargeService.Instance.Request(eventType, mat, intensity, "default", phys, sig,
                clip => { if (src != null) src.PlayOneShot(clip, vol); }, null);
        }

        public string IntensityFor(float v)
        {
            return v >= hardVelocity ? "hard" : (v <= softVelocity ? "soft" : "normal");
        }

        public float SpeedGain(float vn)
        {
            float centre = Mathf.Sqrt(Mathf.Max(1e-3f, softVelocity * hardVelocity));
            float lim = Mathf.Pow(10f, speedGainMaxDb / 20f);
            return Mathf.Clamp(Mathf.Pow(Mathf.Max(vn, 1e-3f) / centre, 1.2f), 1f / lim, lim);
        }

        static float NormalSpeed(Collision c)
        {
            if (c.contactCount == 0) return c.relativeVelocity.magnitude;
            return Mathf.Abs(Vector3.Dot(c.relativeVelocity, c.GetContact(0).normal));
        }

        public static string MaterialOf(Collider col)
        {
            if (col == null) return null;
            var surf = col.GetComponentInParent<SargeSurface>();
            if (surf != null && !string.IsNullOrEmpty(surf.material)) return surf.material;
            var pm = col.sharedMaterial;
            if (pm != null && !string.IsNullOrEmpty(pm.name))
                return pm.name.Replace(" (Instance)", "").Trim().ToLower();
            return null;
        }
    }
}
