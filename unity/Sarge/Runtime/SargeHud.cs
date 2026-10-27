using System;
using UnityEngine;

namespace Sarge
{
    public static class SargeHud
    {
        public static bool Visible;
        public static KeyCode Key = KeyCode.F1;
        static int _frame = -1;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetOnPlay()
        {
            Visible = false;
            _frame = -1;
        }

        public static void Tick()
        {
            if (_frame == Time.frameCount) return;
            _frame = Time.frameCount;
            bool down;
            try { down = Input.GetKeyDown(Key); }
            catch (InvalidOperationException) { return; }
            if (down) Visible = !Visible;
        }
    }
}
