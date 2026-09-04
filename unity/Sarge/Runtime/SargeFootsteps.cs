using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(AudioSource))]
    public class SargeFootsteps : MonoBehaviour
    {
        public AudioSource source;
        public string equipped = null;
        public string defaultMaterial = "grass";
        public float interval = 0.5f;
        public float rayDistance = 2f;
        public bool autoStep = true;
        public AudioClip authoredFallback;

        float _t;
        string _surface;

        void Reset()
        {
            source = GetComponent<AudioSource>();
        }

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        void Start()
        {
            _surface = DetectSurface();
            PushContext();
        }

        void Update()
        {
            string s = DetectSurface();
            if (s != _surface)
            {
                _surface = s;
                PushContext();
            }
            if (!autoStep) return;
            _t += Time.deltaTime;
            if (_t >= interval)
            {
                _t = 0f;
                Step();
            }
        }

        public void Step()
        {
            string mat = _surface ?? defaultMaterial;
            var src = source;
            SargeService.Instance.Request("footstep", mat, "normal", "default",
                clip => { if (src != null) src.PlayOneShot(clip); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); });
        }

        void PushContext()
        {
            SargeService.Instance.SetContext(equipped, new[] { _surface ?? defaultMaterial }, "default");
        }

        string DetectSurface()
        {
            RaycastHit hit;
            if (Physics.Raycast(transform.position + Vector3.up * 0.1f, Vector3.down, out hit, rayDistance))
            {
                var surf = hit.collider.GetComponentInParent<SargeSurface>();
                if (surf != null && !string.IsNullOrEmpty(surf.material)) return surf.material;
            }
            return defaultMaterial;
        }
    }
}
