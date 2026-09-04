using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(AudioSource))]
    public class SargeWalker : MonoBehaviour
    {
        public AudioSource source;
        public float moveSpeed = 4f;
        public float strideLength = 1.6f;
        public float rayDistance = 3f;
        public string defaultMaterial = "grass";
        public string[] arenaSurfaces = { "grass", "stone", "metal", "wood" };
        public string equipped = null;
        public AudioClip authoredFallback;

        float _dist;
        string _surface;

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        void Start()
        {
            _surface = DetectSurface();
            SargeService.Instance.SetContext(equipped, arenaSurfaces, "arena");
        }

        void Update()
        {
            float h = Input.GetAxisRaw("Horizontal");
            float v = Input.GetAxisRaw("Vertical");
            var move = new Vector3(h, 0f, v);
            if (move.sqrMagnitude > 1f) move.Normalize();
            var delta = move * (moveSpeed * Time.deltaTime);
            transform.position += delta;

            float moved = delta.magnitude;
            if (moved <= 0.00001f) return;
            _surface = DetectSurface();
            _dist += moved;
            if (_dist >= strideLength)
            {
                _dist -= strideLength;
                Step();
            }
        }

        void Step()
        {
            string mat = _surface ?? defaultMaterial;
            var src = source;
            SargeService.Instance.Request("footstep", mat, "normal", "default",
                c => { if (src != null) src.PlayOneShot(c); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); });
        }

        string DetectSurface()
        {
            RaycastHit hit;
            if (Physics.Raycast(transform.position + Vector3.up * 0.5f, Vector3.down, out hit, rayDistance))
            {
                var s = hit.collider.GetComponentInParent<SargeSurface>();
                if (s != null && !string.IsNullOrEmpty(s.material)) return s.material;
            }
            return defaultMaterial;
        }

        void OnGUI()
        {
            var svc = SargeService.Instance;
            GUI.Label(new Rect(12, 10, 460, 22), "WASD / arrows to move  ·  stand still = silence");
            GUI.Label(new Rect(12, 30, 460, 22), "on: " + (_surface ?? "-") + "    served: " + (svc.LastServed ?? "-") +
                "    cached " + svc.CachedCount + " / fallback " + svc.FallbackCount + (svc.IsConnected ? "" : "    [OFFLINE]"));
        }
    }
}
