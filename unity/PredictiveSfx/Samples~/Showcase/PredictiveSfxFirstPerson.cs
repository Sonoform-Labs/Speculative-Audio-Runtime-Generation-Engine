using UnityEngine;

namespace PredictiveSfx
{
    [RequireComponent(typeof(AudioSource))]
    public class PredictiveSfxFirstPerson : MonoBehaviour
    {
        public AudioSource source;
        public Camera cam;
        public float moveSpeed = 4f;
        public float mouseSensitivity = 2f;
        public float strideLength = 1.8f;
        public float rayDistance = 3f;
        public string defaultMaterial = "grass";
        public string[] arenaSurfaces = { "grass", "stone", "metal", "wood" };
        public string equipped = null;
        public AudioClip authoredFallback;

        float _dist, _yaw, _pitch;
        string _surface;

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        void Start()
        {
            if (cam == null) cam = GetComponentInChildren<Camera>();
            _surface = DetectSurface();
            PredictiveSfxService.Instance.SetContext(equipped, arenaSurfaces, "arena", 0.3f);
            Cursor.lockState = CursorLockMode.Locked;
            _yaw = transform.eulerAngles.y;
        }

        void Update()
        {
            if (!PredictiveSfxService.Instance.IsConnected) return;
            _yaw += Input.GetAxisRaw("Mouse X") * mouseSensitivity;
            _pitch = Mathf.Clamp(_pitch - Input.GetAxisRaw("Mouse Y") * mouseSensitivity, -80f, 80f);
            transform.rotation = Quaternion.Euler(0f, _yaw, 0f);
            if (cam != null) cam.transform.localRotation = Quaternion.Euler(_pitch, 0f, 0f);

            float h = Input.GetAxisRaw("Horizontal");
            float v = Input.GetAxisRaw("Vertical");
            var dir = transform.right * h + transform.forward * v;
            if (dir.sqrMagnitude > 1f) dir.Normalize();
            var delta = dir * (moveSpeed * Time.deltaTime);
            transform.position += delta;

            if (Input.GetKeyDown(KeyCode.Escape)) Cursor.lockState = CursorLockMode.None;
            if (Input.GetMouseButtonDown(0)) Cursor.lockState = CursorLockMode.Locked;

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
            PredictiveSfxService.Instance.Request("footstep", mat, "normal", "default",
                c => { if (src != null) src.PlayOneShot(c); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); });
        }

        string DetectSurface()
        {
            RaycastHit hit;
            if (Physics.Raycast(transform.position + Vector3.up * 0.5f, Vector3.down, out hit, rayDistance))
            {
                var s = hit.collider.GetComponentInParent<PredictiveSfxSurface>();
                if (s != null && !string.IsNullOrEmpty(s.material)) return s.material;
            }
            return defaultMaterial;
        }

        void OnGUI()
        {
            var svc = PredictiveSfxService.Instance;
            if (!svc.IsConnected)
            {
                float w = 460f, h = 64f;
                GUI.Box(new Rect((Screen.width - w) / 2f, (Screen.height - h) / 2f, w, h),
                    "Starting sound engine...\nfirst run can take ~30s (loading the neural model)");
                return;
            }
            GUI.Label(new Rect(12, 10, 540, 22), "WASD move  ·  mouse look  ·  Esc free cursor / click to relock");
            GUI.Label(new Rect(12, 30, 540, 22), "on: " + (_surface ?? "-") + "    served: " + (svc.LastServed ?? "-") +
                "    cached " + svc.CachedCount + " / fallback " + svc.FallbackCount + (svc.IsConnected ? "" : "    [OFFLINE]"));
            if (svc.CachedCount + svc.FallbackCount == 0)
                GUI.Label(new Rect(12, 54, 540, 22), "warming up the sound palette (generating all surfaces)... give it a few seconds");
        }
    }
}
