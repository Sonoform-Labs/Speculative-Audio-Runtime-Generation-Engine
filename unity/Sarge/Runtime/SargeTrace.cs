using System.Collections.Generic;
using System.Threading;

namespace Sarge
{
    public struct SargeTraceRecord
    {
        public long seq;
        public string kind;
        public string eventType;
        public string material;
        public string intensity;
        public string path;
        public string route;
        public string gen;
        public string prompt;
        public string take;
        public string source;
        public string detail;
        public bool phys;
        public bool scene;
        public bool physSent;
        public bool sceneSent;
        public bool predicted;
        public float ms;
        public float eta;
        public int count;
        public int queued;
        public int variant;
    }

    public static class SargeTrace
    {
        public const int Capacity = 512;
        static readonly SargeTraceRecord[] _ring = new SargeTraceRecord[Capacity];
        static readonly object _lock = new object();
        static long _seq;
        static int _listeners;

        public static volatile string Generator;
        public static volatile string UpgradeState;
        public static volatile string WriterState;
        public static volatile bool Connected;

        public static bool Listening => Volatile.Read(ref _listeners) > 0;

        public static void Listen() { Interlocked.Increment(ref _listeners); }

        public static void Unlisten()
        {
            if (Interlocked.Decrement(ref _listeners) < 0) Interlocked.Exchange(ref _listeners, 0);
        }

        public static long Latest
        {
            get { lock (_lock) return _seq; }
        }

        public static void Add(SargeTraceRecord r)
        {
            lock (_lock)
            {
                r.seq = ++_seq;
                _ring[r.seq % Capacity] = r;
            }
        }

        public static long Since(long after, List<SargeTraceRecord> into)
        {
            lock (_lock)
            {
                long from = after < _seq - Capacity + 1 ? _seq - Capacity + 1 : after + 1;
                if (from < 1) from = 1;
                for (long s = from; s <= _seq; s++) into.Add(_ring[s % Capacity]);
                return _seq;
            }
        }

        public static void Clear()
        {
            lock (_lock)
            {
                _seq = 0;
                for (int i = 0; i < Capacity; i++) _ring[i] = default;
            }
        }
    }
}
