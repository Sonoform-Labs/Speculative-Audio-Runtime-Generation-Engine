using System;
using System.Globalization;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

namespace PredictiveSfx
{
    [Serializable]
    public class Welcome
    {
        public string type;
        public int proto;
        public string server_version;
        public string generator;
        public int sample_rate;
        public int channels;
        public string pcm;
        public long max_pcm_bytes;
    }

    [Serializable]
    public class AudioMeta
    {
        public string type;
        public string served;
        public int variant_id;
        public string format;
        public int sample_rate;
        public int channels;
        public int n_samples;
        public int byte_len;
        public float duration_s;
        public string intent;
    }

    [Serializable]
    public class ErrorMeta
    {
        public string type;
        public string code;
        public string message;
        public bool fatal;
    }

    public class AudioResult
    {
        public AudioMeta meta;
        public float[] samples;
    }

    public struct SfxPrediction
    {
        public string source;
        public string eventType;
        public string material;
        public string intensity;
        public string context;
        public float etaS;
        public float confidence;
    }

    public class SfxProtocolException : Exception
    {
        public string Code;
        public bool Fatal;

        public SfxProtocolException(string code, string message, bool fatal) : base(code + ": " + message)
        {
            Code = code;
            Fatal = fatal;
        }
    }

    public class PredictiveSfxClient
    {
        const byte Version = 1;
        const int HeaderSize = 16;
        const int MaxJson = 65536;
        const int MaxPcm = 4194304;
        static readonly byte[] Magic = Encoding.ASCII.GetBytes("PSFX");

        const byte HELLO = 1, WELCOME = 2, CONTEXT = 3, CONTEXT_ACK = 4, EVENT = 5, AUDIO = 6,
                   PREBAKE = 7, PREBAKE_ACK = 8, STATS = 9, STATS_REPLY = 10, PING = 11, PONG = 12,
                   PREDICT = 13, PREDICT_ACK = 14, ERROR = 255;

        TcpClient _tcp;
        NetworkStream _stream;
        string _pcm = "f32le";

        public Welcome Welcome { get; private set; }
        public int SampleRate { get; private set; } = 48000;

        public void Connect(string host = "127.0.0.1", int port = 8765, string pcm = "f32le")
        {
            _pcm = pcm;
            _tcp = new TcpClient { NoDelay = true };
            _tcp.Connect(host, port);
            _stream = _tcp.GetStream();
            Hello();
        }

        public void Close()
        {
            if (_stream != null) { _stream.Close(); _stream = null; }
            if (_tcp != null) { _tcp.Close(); _tcp = null; }
        }

        void Hello()
        {
            WriteFrame(HELLO, "{\"type\":\"hello\",\"proto\":1,\"client\":\"unity\",\"pcm\":" + Quote(_pcm) + "}");
            var f = ReadFrame();
            Welcome = JsonUtility.FromJson<Welcome>(f.json);
            _pcm = Welcome.pcm;
            SampleRate = Welcome.sample_rate;
        }

        public void Context(float timeS, string equipped, string[] surfaces, string region = "default", float gpuPressure = 1f)
        {
            var sb = new StringBuilder();
            sb.Append("{\"type\":\"context\",\"time_s\":").Append(F(timeS));
            sb.Append(",\"equipped\":").Append(equipped == null ? "null" : Quote(equipped));
            sb.Append(",\"surfaces\":[");
            if (surfaces != null)
                for (int i = 0; i < surfaces.Length; i++) { if (i > 0) sb.Append(','); sb.Append(Quote(surfaces[i])); }
            sb.Append("],\"region\":").Append(Quote(region));
            sb.Append(",\"gpu_pressure\":").Append(F(gpuPressure)).Append('}');
            WriteFrame(CONTEXT, sb.ToString());
            ReadFrame();
        }

        public AudioResult Event(string eventType, string material, string intensity = "normal", string context = "default")
        {
            var json = "{\"type\":\"event\",\"event_type\":" + Quote(eventType) + ",\"material\":" + Quote(material)
                + ",\"intensity\":" + Quote(intensity) + ",\"context\":" + Quote(context) + "}";
            WriteFrame(EVENT, json);
            var f = ReadFrame();
            var meta = JsonUtility.FromJson<AudioMeta>(f.json);
            return new AudioResult { meta = meta, samples = DecodePcm(f.pcm, meta.format, meta.n_samples) };
        }

        public AudioClip EventClip(string eventType, string material, string intensity = "normal", string context = "default")
        {
            var r = Event(eventType, material, intensity, context);
            var clip = AudioClip.Create(eventType + "_" + material, r.samples.Length, 1, r.meta.sample_rate, false);
            clip.SetData(r.samples, 0);
            return clip;
        }

        public void Ping(float t)
        {
            WriteFrame(PING, "{\"type\":\"ping\",\"t\":" + F(t) + "}");
            ReadFrame();
        }

        public void Predict(System.Collections.Generic.List<SfxPrediction> preds, float now)
        {
            var sb = new StringBuilder();
            sb.Append("{\"type\":\"predict\",\"now\":").Append(F(now)).Append(",\"predictions\":[");
            for (int i = 0; i < preds.Count; i++)
            {
                var p = preds[i];
                if (i > 0) sb.Append(',');
                sb.Append("{\"source\":").Append(Quote(p.source ?? "motion"));
                sb.Append(",\"event_type\":").Append(Quote(p.eventType));
                sb.Append(",\"material\":").Append(Quote(p.material));
                sb.Append(",\"intensity\":").Append(Quote(p.intensity ?? "normal"));
                sb.Append(",\"context\":").Append(Quote(p.context ?? "default"));
                sb.Append(",\"eta_s\":").Append(F(p.etaS));
                sb.Append(",\"confidence\":").Append(F(p.confidence)).Append('}');
            }
            sb.Append("]}");
            WriteFrame(PREDICT, sb.ToString());
            ReadFrame();
        }

        static float[] DecodePcm(byte[] buf, string format, int n)
        {
            var outp = new float[n];
            if (format == "f32le") { Buffer.BlockCopy(buf, 0, outp, 0, n * 4); return outp; }
            if (format == "i16le")
            {
                for (int i = 0; i < n; i++) outp[i] = BitConverter.ToInt16(buf, i * 2) / 32767f;
                return outp;
            }
            throw new Exception("unknown pcm format " + format);
        }

        static string Quote(string s) => "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        static string F(float v) => v.ToString(CultureInfo.InvariantCulture);

        struct Frame { public byte msgType; public string json; public byte[] pcm; }

        void WriteFrame(byte msgType, string json, byte[] pcm = null)
        {
            var jb = Encoding.UTF8.GetBytes(json);
            int pcmLen = pcm == null ? 0 : pcm.Length;
            var frame = new byte[HeaderSize + jb.Length + pcmLen];
            Buffer.BlockCopy(Magic, 0, frame, 0, 4);
            frame[4] = Version;
            frame[5] = msgType;
            frame[6] = (byte)(pcmLen > 0 ? 1 : 0);
            frame[7] = 0;
            Buffer.BlockCopy(BitConverter.GetBytes((uint)jb.Length), 0, frame, 8, 4);
            Buffer.BlockCopy(BitConverter.GetBytes((uint)pcmLen), 0, frame, 12, 4);
            Buffer.BlockCopy(jb, 0, frame, HeaderSize, jb.Length);
            if (pcmLen > 0) Buffer.BlockCopy(pcm, 0, frame, HeaderSize + jb.Length, pcmLen);
            _stream.Write(frame, 0, frame.Length);
        }

        Frame ReadFrame()
        {
            var head = ReadExactly(HeaderSize);
            for (int i = 0; i < 4; i++) if (head[i] != Magic[i]) throw new Exception("bad magic");
            if (head[4] != Version) throw new Exception("unsupported version " + head[4]);
            uint jsonLen = BitConverter.ToUInt32(head, 8);
            uint pcmLen = BitConverter.ToUInt32(head, 12);
            if (jsonLen > MaxJson || pcmLen > MaxPcm) throw new Exception("oversized frame");
            var jb = ReadExactly((int)jsonLen);
            var pcm = pcmLen > 0 ? ReadExactly((int)pcmLen) : Array.Empty<byte>();
            byte msgType = head[5];
            if (msgType == ERROR)
            {
                var e = JsonUtility.FromJson<ErrorMeta>(Encoding.UTF8.GetString(jb));
                throw new SfxProtocolException(e.code, e.message, e.fatal);
            }
            return new Frame { msgType = msgType, json = Encoding.UTF8.GetString(jb), pcm = pcm };
        }

        byte[] ReadExactly(int n)
        {
            var buf = new byte[n];
            int got = 0;
            while (got < n)
            {
                int r = _stream.Read(buf, got, n - got);
                if (r == 0) throw new Exception("peer closed");
                got += r;
            }
            return buf;
        }
    }
}
