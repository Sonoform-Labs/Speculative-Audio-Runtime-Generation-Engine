using System.Diagnostics;
using UnityEngine;

namespace Sarge
{
    public class SargeLauncher : MonoBehaviour
    {
        public bool autoStart = true;
        public string pythonExe = @"D:\Python\SARGE\.venv\Scripts\python.exe";
        public string workingDir = @"D:\Python\SARGE";
        public string arguments = @"scripts\run_sidecar.py --generator sa3 --tier default --realtime --learn";
        public string hfHome = @"D:\hf_cache";
        public string host = "127.0.0.1";
        public int port = 8765;

        static SargeLauncher _instance;
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
            string logPath = System.IO.Path.Combine(workingDir, "sidecar.log");
            if (AlreadyRunning())
            {
                UnityEngine.Debug.Log("[PSFX] a sidecar is already running on " + host + ":" + port + "; reusing it");
                return;
            }
            if (!System.IO.File.Exists(pythonExe))
            {
                UnityEngine.Debug.LogWarning("[PSFX] python not found at '" + pythonExe +
                    "'. Fix the paths on the Sarge Launcher component, or start the sidecar manually.");
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
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                if (!string.IsNullOrEmpty(hfHome))
                {
                    psi.EnvironmentVariables["HF_HOME"] = hfHome;
                    psi.EnvironmentVariables["HF_HUB_OFFLINE"] = "1";
                }
                try { System.IO.File.WriteAllText(logPath, "[psfx] launching: " + pythonExe + " " + arguments + "\n(cwd " + workingDir + ")\n"); } catch { }
                _proc = new Process { StartInfo = psi };
                _proc.OutputDataReceived += (s, e) => Append(logPath, e.Data);
                _proc.ErrorDataReceived += (s, e) => Append(logPath, e.Data);
                _proc.Start();
                _proc.BeginOutputReadLine();
                _proc.BeginErrorReadLine();
                _spawned = true;
                UnityEngine.Debug.Log("[PSFX] launched sidecar; warming model (~30s). Output -> " + logPath);
            }
            catch (System.Exception e)
            {
                UnityEngine.Debug.LogWarning("[PSFX] could not launch sidecar (" + e.Message +
                    "). Check the paths on the Sarge Launcher, or run the sidecar manually.");
            }
        }

        static readonly object _logLock = new object();

        static void Append(string path, string line)
        {
            if (string.IsNullOrEmpty(line)) return;
            try { lock (_logLock) { System.IO.File.AppendAllText(path, line + "\n"); } } catch { }
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
