using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(AudioSource))]
    public class SargeFootsteps : MonoBehaviour
    {
        public AudioSource source;
        public string eventType = "footstep";
        public string equipped = null;
        public string defaultMaterial = "";
        public string context = "";
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
            var svc = SargeService.Instance;
            svc.Request(eventType, mat ?? "", "normal", svc.ContextOr(context), null, null,
                clip => { if (src != null) src.PlayOneShot(clip); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); },
                null, null, svc.sendScene ? SargeScene.ForActor(this, "step") : null);
        }

        void PushContext()
        {
            string mat = _surface ?? defaultMaterial;
            var svc = SargeService.Instance;
            svc.SetContext(equipped, string.IsNullOrEmpty(mat) ? new string[0] : new[] { mat }, svc.ContextOr(context));
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
