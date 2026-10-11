import os
import shutil
import subprocess
import sys
import time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tag = argv[0] if argv else "compiled"
    extra = argv[1:]
    out = os.path.join(BENCH, "out_e2e_sidecar.json")
    bak = out + ".bak"
    if os.path.isfile(out):
        shutil.copy(out, bak)
    log_path = os.path.join(BENCH, "e2e_%s_server.log" % tag)
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    srv = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "run_sidecar.py"), "--port", "8766",
                            "--tier", "default"] + extra, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    startup = None
    try:
        while time.time() - t0 < 400:
            if srv.poll() is not None:
                break
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                if "sidecar READY" in fh.read():
                    startup = time.time() - t0
                    break
            time.sleep(0.5)
        print("startup to READY: %s s" % (None if startup is None else round(startup, 1)), flush=True)
        if startup is not None:
            subprocess.run([sys.executable, "-u", os.path.join(BENCH, "e2e_sidecar.py"), "8766"], cwd=ROOT,
                           timeout=900)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()
    if os.path.isfile(out):
        dst = os.path.join(BENCH, "out_e2e_sidecar_%s.json" % tag)
        shutil.move(out, dst)
        with open(dst, "a", encoding="utf-8") as fh:
            fh.write("\n")
        print("saved -> %s (startup %s s)" % (dst, None if startup is None else round(startup, 1)))
    if os.path.isfile(bak):
        shutil.move(bak, out)
    return 0 if startup is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
