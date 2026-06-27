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

        public bool showLandMarker = true;

        float _dist, _yaw, _pitch, _groundY, _vy;
        bool _grounded = true, _hasGun;
        int _ammo;
        string _surface;
        GameObject _landMarker;
        string _predLandSurface;

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
            if (!_grounded) PredictLanding();
            else { _predLandSurface = null; if (_landMarker != null) _landMarker.SetActive(false); }

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

        void PredictLanding()
        {
            float height = transform.position.y - _groundY;
            if (height <= 0.01f) return;
            float t = (_vy + Mathf.Sqrt(_vy * _vy + 2f * gravity * height)) / gravity;
            if (t <= 0.01f) return;
            float h = Input.GetAxisRaw("Horizontal");
            float v = Input.GetAxisRaw("Vertical");
            var dir = transform.right * h + transform.forward * v;
            if (dir.sqrMagnitude > 1f) dir.Normalize();
            Vector3 land = transform.position + dir * (moveSpeed * t);
            Vector3 hitPoint;
            string surf = SurfaceAtPoint(land, out hitPoint);
            if (surf != _predLandSurface)
            {
                _predLandSurface = surf;
                PredictiveSfxService.Instance.SendPredictions(
                    new System.Collections.Generic.List<SfxPrediction>(1)
                    {
                        new SfxPrediction { source = "motion", eventType = "land", material = surf,
                            intensity = "normal", context = "default", etaS = t, confidence = 1f },
                    }, Time.time);
            }
            UpdateLandMarker(hitPoint);
        }

        string SurfaceAtPoint(Vector3 p, out Vector3 hitPoint)
        {
            hitPoint = new Vector3(p.x, _groundY, p.z);
            RaycastHit hit;
            if (Physics.Raycast(new Vector3(p.x, _groundY + 5f, p.z), Vector3.down, out hit, 10f))
            {
                hitPoint = hit.point;
                var s = hit.collider.GetComponentInParent<PredictiveSfxSurface>();
                if (s != null && !string.IsNullOrEmpty(s.material)) return s.material;
            }
            return defaultMaterial;
        }

        void UpdateLandMarker(Vector3 p)
        {
            if (!showLandMarker) return;
            if (_landMarker == null)
            {
                _landMarker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                var col = _landMarker.GetComponent<Collider>();
                if (col != null) Destroy(col);
                _landMarker.layer = 2;
                _landMarker.name = "psfx_landing_marker";
                _landMarker.transform.localScale = new Vector3(0.7f, 0.02f, 0.7f);
                var mr = _landMarker.GetComponent<Renderer>();
                var c = new Color(0.2f, 0.7f, 1f);
                mr.material.color = c;
                if (mr.material.HasProperty("_BaseColor")) mr.material.SetColor("_BaseColor", c);
                if (mr.material.HasProperty("_EmissionColor"))
                {
                    mr.material.EnableKeyword("_EMISSION");
                    mr.material.SetColor("_EmissionColor", c * 0.8f);
                }
            }
            _landMarker.SetActive(true);
            _landMarker.transform.position = p + Vector3.up * 0.03f;
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
            GUI.Label(new Rect(12, 10, 820, 22), "engine: " + (svc.Generator ?? "?") + "    on: " + (_surface ?? "-") + "    gun: " + (_hasGun ? "ammo " + _ammo : "holstered") +
                "    served: " + (svc.LastServed ?? "-") + "    cached " + svc.CachedCount + " / realtime " + svc.RealtimeCount + " / fallback " + svc.FallbackCount);
            GUI.Label(new Rect(12, 34, 700, 22), "predicting ahead -> " + (svc.LastPredicted ?? "(none yet)") +
                "    (sent " + svc.PredictionsSent + ")");
            GUI.Label(new Rect(12, 58, 700, 22), "intent (habit guess) -> " +
                (string.IsNullOrEmpty(svc.LastIntent) ? "(learning your habits...)" : svc.LastIntent));
        }
    }
}
