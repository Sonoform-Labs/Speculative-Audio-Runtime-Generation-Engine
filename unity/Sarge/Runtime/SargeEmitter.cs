using UnityEngine;

namespace Sarge
{
    [RequireComponent(typeof(AudioSource))]
    public class SargeEmitter : MonoBehaviour
    {
        public AudioSource source;
        public string eventType = "";
        public string material = "";
        public string intensity = "normal";
        public string context = "";
        public string eventClass = "";
        public string prompt = "";
        public string madeBy = "";
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
            if (string.IsNullOrEmpty(ev))
            {
                Debug.LogWarning("[SARGE] " + name + ": SargeEmitter has no event type");
                return;
            }
            var svc = SargeService.Instance;
            svc.Request(ev, mat ?? "", intensity, svc.ContextOr(context), null, null,
                clip => { if (src != null) src.PlayOneShot(clip); },
                () => { if (authoredFallback != null && src != null) src.PlayOneShot(authoredFallback); },
                eventClass, prompt, svc.sendScene ? SargeScene.ForEmitter(this, madeBy) : null);
        }
    }
}
