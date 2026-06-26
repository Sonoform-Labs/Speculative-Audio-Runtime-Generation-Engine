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
        public float jumpSpeed = 5f;
        public float gravity = 14f;
        public int magazine = 8;
        public bool startWithGun = true;
        public string defaultMaterial = "grass";
        public string[] arenaSurfaces = { "grass", "stone", "metal", "wood" };
        public AudioClip authoredFallback;

        float _dist, _yaw, _pitch, _groundY, _vy;
        bool _grounded = true, _hasGun;
        int _ammo;
        string _surface;

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        void Start()
        {
            if (cam == null) cam = GetComponentInChildren<Camera>();
            _groundY = transform.position.y;
            _hasGun = startWithGun;
            _ammo = magazine;
            _surface = DetectSurface();
            PushContext();
            Cursor.lockState = CursorLockMode.Locked;
            _yaw = transform.eulerAngles.y;
        }

        void PushContext()
        {
            PredictiveSfxService.Instance.SetContext(_hasGun ? "gun" : null, arenaSurfaces, "arena", 0.3f);
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

            if (_grounded && Input.GetKeyDown(KeyCode.Space))
            {
                _vy = jumpSpeed;
                _grounded = false;
                Fire("jump", _surface);
            }
            _vy -= gravity * Time.deltaTime;
            float ny = transform.position.y + _vy * Time.deltaTime;
            bool landed = false;
            if (ny <= _groundY)
            {
                ny = _groundY;
                if (!_grounded) landed = true;
                _grounded = true;
                _vy = 0f;
            }
            transform.position = new Vector3(transform.position.x + delta.x, ny, transform.position.z + delta.z);
            if (landed)
            {
                _surface = DetectSurface();
                Fire("land", _surface);
            }

            if (Input.GetKeyDown(KeyCode.G)) { _hasGun = !_hasGun; PushContext(); }
            if (Input.GetMouseButtonDown(0))
            {
                if (Cursor.lockState != CursorLockMode.Locked) Cursor.lockState = CursorLockMode.Locked;
                else if (_hasGun)
                {
                    if (_ammo > 0) { _ammo--; Fire("shoot", "metal"); }
                    else Fire("empty_click", "metal");
                }
            }
            if (_hasGun && Input.GetKeyDown(KeyCode.R)) { _ammo = magazine; Fire("reload", "metal"); }
            if (Input.GetKeyDown(KeyCode.Escape)) Cursor.lockState = CursorLockMode.None;

            if (!_grounded) return;
            float moved = new Vector2(delta.x, delta.z).magnitude;
            if (moved <= 0.00001f) return;
            _surface = DetectSurface();
            _dist += moved;
            if (_dist >= strideLength)
            {
                _dist -= strideLength;
                Fire("footstep", _surface);
            }
        }

        void Fire(string ev, string mat)
        {
            var src = source;
            PredictiveSfxService.Instance.Request(ev, mat ?? defaultMaterial, "normal", "default",
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
                float w = 460f, hh = 64f;
                GUI.Box(new Rect((Screen.width - w) / 2f, (Screen.height - hh) / 2f, w, hh),
                    "Starting sound engine...\nfirst run can take ~30s (loading the neural model)");
                return;
            }
            GUI.Label(new Rect(12, 10, 620, 22), "WASD move  ·  mouse look  ·  Space jump  ·  click fire  ·  R reload  ·  G holster  ·  Esc cursor");
            GUI.Label(new Rect(12, 30, 620, 22), "on: " + (_surface ?? "-") + "    gun: " + (_hasGun ? "ammo " + _ammo : "holstered") +
                "    served: " + (svc.LastServed ?? "-") + "    cached " + svc.CachedCount + " / fallback " + svc.FallbackCount);
            GUI.Label(new Rect(12, 54, 700, 22), "predicting ahead -> " + (svc.LastPredicted ?? "(none yet)") +
                "    (sent " + svc.PredictionsSent + ")");
            if (svc.CachedCount + svc.FallbackCount == 0)
                GUI.Label(new Rect(12, 78, 620, 22), "warming up the sound palette...");
        }
    }
}
