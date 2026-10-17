using UnityEngine;

namespace Sarge
{
    public static class SargePlayback
    {
        public const int PoolSize = 16;
        public static float Spread = 60f;

        static GameObject _root;
        static AudioSource[] _pool;
        static float[] _busyUntil;
        static int _next;

        public static AudioSource PlayAt(AudioClip c, Vector3 p, float volume, float minDistance, float maxDistance, int priority)
        {
            if (c == null) return null;
            Ensure();
            int i = Pick();
            var s = _pool[i];
            s.Stop();
            s.transform.position = p;
            s.clip = c;
            s.volume = volume;
            s.minDistance = Mathf.Max(0.01f, minDistance);
            s.maxDistance = Mathf.Max(s.minDistance, maxDistance);
            s.priority = Mathf.Clamp(priority, 0, 256);
            s.spread = Spread;
            s.Play();
            _busyUntil[i] = Time.unscaledTime + c.length / Mathf.Max(0.01f, Mathf.Abs(s.pitch));
            return s;
        }

        static void Ensure()
        {
            if (_root != null) return;
            _root = new GameObject("SargePlayback") { hideFlags = HideFlags.HideInHierarchy };
            Object.DontDestroyOnLoad(_root);
            _pool = new AudioSource[PoolSize];
            _busyUntil = new float[PoolSize];
            _next = 0;
            for (int i = 0; i < PoolSize; i++)
            {
                var go = new GameObject("SargePlayback voice " + i) { hideFlags = HideFlags.HideInHierarchy };
                go.transform.SetParent(_root.transform, false);
                var s = go.AddComponent<AudioSource>();
                s.playOnAwake = false;
                s.spatialBlend = 1f;
                s.rolloffMode = AudioRolloffMode.Logarithmic;
                s.dopplerLevel = 0f;
                s.spread = Spread;
                _pool[i] = s;
            }
        }

        static int Pick()
        {
            float now = Time.unscaledTime;
            for (int k = 0; k < PoolSize; k++)
            {
                int i = (_next + k) % PoolSize;
                if (_busyUntil[i] <= now || !_pool[i].isPlaying)
                {
                    _next = (i + 1) % PoolSize;
                    return i;
                }
            }
            int oldest = 0;
            for (int i = 1; i < PoolSize; i++)
                if (_busyUntil[i] < _busyUntil[oldest]) oldest = i;
            _next = (oldest + 1) % PoolSize;
            return oldest;
        }
    }
}
