using System.Diagnostics;
using UnityEngine;

namespace PredictiveSfx
{
    public class PredictiveSfxLauncher : MonoBehaviour
    {
        public bool autoStart = true;
        public string pythonExe = @"D:\Python\Predictive-Neural-SFX\.venv\Scripts\python.exe";
        public string workingDir = @"D:\Python\Predictive-Neural-SFX";
        public string arguments = @"scripts\run_sidecar.py --generator sa3";
        public string hfHome = @"D:\hf_cache";
        public string host = "127.0.0.1";
        public int port = 8765;

        static PredictiveSfxLauncher _instance;
        Process _proc;
        bool _spawned;

        void Awake()
        {
            if (_instance != null && _instance != this) { Destroy(gameObject); return; }
            _instance = this;
            DontDestroyOnLoad(gameObject);
            if (autoStart) StartSidecar();
        }

        void StartSidecar()
        {
            if (AlreadyRunning())
            {
                UnityEngine.Debug.Log("[PSFX] sidecar already running; reusing it");
                return;
            }
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = arguments,
                    WorkingDirectory = workingDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                if (!string.IsNullOrEmpty(hfHome))
                {
                    psi.EnvironmentVariables["HF_HOME"] = hfHome;
                    psi.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
                }
                _proc = Process.Start(psi);
                _spawned = true;
                UnityEngine.Debug.Log("[PSFX] launched sidecar; gameplay waits until it is ready");
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogWarning("[PSFX] could not launch sidecar (" + e.Message +
                    "). Set the python/working-dir paths on the Predictive Sfx Launcher, or start it manually.");
            }
        }

        bool AlreadyRunning()
        {
            try
            {
                using (var t = new System.Net.Sockets.TcpClient())
                {
                    var ar = t.BeginConnect(host, port, null, null);
                    if (ar.AsyncWaitHandle.WaitOne(300)) { t.EndConnect(ar); return true; }
                }
            }
            catch { }
            return false;
        }

        void Shutdown()
        {
            if (_spawned && _proc != null)
            {
                try { if (!_proc.HasExited) { _proc.Kill(); _proc.WaitForExit(2000); } } catch { }
            }
            _proc = null;
        }

        void OnApplicationQuit() { Shutdown(); }
        void OnDestroy() { if (_instance == this) Shutdown(); }
    }
}
