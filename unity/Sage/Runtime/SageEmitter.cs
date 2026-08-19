using UnityEngine;

namespace Sage
{
    [RequireComponent(typeof(AudioSource))]
    public class SageEmitter : MonoBehaviour
    {
        public AudioSource source;
        public string eventType = "shoot";
        public string material = "metal";
        public string intensity = "normal";
        public AudioClip authoredFallback;

        void Reset()
        {
            source = GetComponent<AudioSource>();
        }

        void Awake()
        {
            if (source == null) source = GetComponent<AudioSource>();
        }

        public void Emit()
        {
            Emit(eventType, material);
        }

        public void Emit(string ev, string mat)
        {
            var src = source != null ? source : GetComponent<AudioSource>();
            SageService.Instance.Request(ev, mat, intensity, "default",
                clip => { if (src != null) src.PlayOneShot(clip); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); });
        }
    }
}
