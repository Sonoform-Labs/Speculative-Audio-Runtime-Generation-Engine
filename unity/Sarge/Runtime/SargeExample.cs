using UnityEngine;

namespace Sarge
{
    public class SargeExample : MonoBehaviour
    {
        public AudioSource source;
        public float footstepInterval = 0.5f;
        public float shootInterval = 2.0f;

        float _footT, _shootT;

        void Start()
        {
            if (source == null) source = gameObject.AddComponent<AudioSource>();
            SargeService.Instance.SetContext("gun", new[] { "grass" }, "field");
        }

        void Update()
        {
            _footT += Time.deltaTime;
            if (_footT >= footstepInterval)
            {
                _footT = 0f;
                SargeService.Instance.Play("footstep", "grass", source);
            }
            _shootT += Time.deltaTime;
            if (_shootT >= shootInterval)
            {
                _shootT = 0f;
                SargeService.Instance.Play("shoot", "metal", source);
            }
        }
    }
}
