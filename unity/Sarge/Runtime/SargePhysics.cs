using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace Sarge
{
    public static class SargePhysics
    {
        static readonly Dictionary<int, string> _bodies = new Dictionary<int, string>();
        static AudioListener _listener;

        public static string Json(Rigidbody a, Collider aCol, Collider b, Vector3 relVel, Vector3 normal, float impulse, float dist)
        {
            return Json(a, aCol, b, relVel, normal, impulse, dist, null);
        }

        public static string Json(Rigidbody a, Collider aCol, Collider b, Vector3 relVel, Vector3 normal, float impulse, float dist, string kind)
        {
            var sb = new StringBuilder(640);
            sb.Append("{\"a\":");
            Body(sb, aCol, a != null ? a : (aCol != null ? aCol.attachedRigidbody : null));
            if (b != null)
            {
                sb.Append(",\"b\":");
                Body(sb, b, b.attachedRigidbody);
            }
            float vn = relVel.magnitude, vt = 0f;
            if (normal.sqrMagnitude > 1e-8f)
            {
                var n = normal.normalized;
                float d = Vector3.Dot(relVel, n);
                vn = Mathf.Abs(d);
                vt = (relVel - d * n).magnitude;
                sb.Append(",\"n\":[").Append(F(n.x)).Append(',').Append(F(n.y)).Append(',').Append(F(n.z)).Append(']');
            }
            sb.Append(",\"vn\":").Append(F(vn)).Append(",\"vt\":").Append(F(vt));
            sb.Append(",\"j\":").Append(F(impulse)).Append(",\"dist\":").Append(F(dist));
            if (!string.IsNullOrEmpty(kind)) sb.Append(",\"kind\":").Append(Quote(kind));
            sb.Append('}');
            return sb.ToString();
        }

        public static string FromCollision(Collision c, Collider self)
        {
            if (c == null || self == null) return null;
            Vector3 normal = Vector3.zero;
            Vector3 point = self.transform.position;
            if (c.contactCount > 0)
            {
                var cp = c.GetContact(0);
                normal = cp.normal;
                point = cp.point;
            }
            return Json(self.attachedRigidbody, self, c.collider, c.relativeVelocity, normal, c.impulse.magnitude,
                ListenerDistance(point));
        }

        public static string Signature(Collider a, Collider b)
        {
            int ia = a != null ? a.GetInstanceID() : 0;
            int ib = b != null ? b.GetInstanceID() : 0;
            return ia.ToString(CultureInfo.InvariantCulture) + ":" + ib.ToString(CultureInfo.InvariantCulture);
        }

        public static float ListenerDistance(Vector3 p)
        {
            if (_listener == null) _listener = Object.FindObjectOfType<AudioListener>();
            return _listener != null ? Vector3.Distance(_listener.transform.position, p) : 0f;
        }

        public static void Forget(Collider c)
        {
            if (c != null) _bodies.Remove(c.GetInstanceID());
        }

        public static void ClearCache()
        {
            _bodies.Clear();
            _listener = null;
        }

        static void Body(StringBuilder sb, Collider col, Rigidbody rb)
        {
            sb.Append(Static(col));
            bool dyn = rb != null && !rb.isKinematic;
            sb.Append(",\"dyn\":").Append(dyn ? '1' : '0');
            if (dyn) sb.Append(",\"m\":").Append(F(rb.mass));
            sb.Append('}');
        }

        static string Static(Collider col)
        {
            if (col == null) return "{\"shape\":\"mesh\"";
            int id = col.GetInstanceID();
            if (_bodies.TryGetValue(id, out var s)) return s;
            Vector3 size;
            string shape = ShapeOf(col, out size);
            var sb = new StringBuilder(256);
            sb.Append("{\"shape\":").Append(Quote(shape));
            sb.Append(",\"ext\":[").Append(F(size.x)).Append(',').Append(F(size.y)).Append(',').Append(F(size.z)).Append(']');
            var pm = col.sharedMaterial;
            if (pm != null)
            {
                sb.Append(",\"pm\":").Append(Quote(Clean(pm.name)));
                sb.Append(",\"bnc\":").Append(F(pm.bounciness)).Append(",\"fr\":").Append(F(pm.dynamicFriction));
            }
            var r = col.GetComponent<Renderer>();
            if (r != null && r.sharedMaterial != null) sb.Append(",\"rm\":").Append(Quote(Clean(r.sharedMaterial.name)));
            var surf = col.GetComponentInParent<SargeSurface>();
            if (surf != null && !string.IsNullOrEmpty(surf.material)) sb.Append(",\"tag\":").Append(Quote(surf.material));
            sb.Append(",\"name\":").Append(Quote(Clean(col.gameObject.name)));
            s = sb.ToString();
            _bodies[id] = s;
            return s;
        }

        static string ShapeOf(Collider col, out Vector3 size)
        {
            var ls = col.transform.lossyScale;
            var sc = new Vector3(Mathf.Abs(ls.x), Mathf.Abs(ls.y), Mathf.Abs(ls.z));
            if (col is SphereCollider sphere)
            {
                float d = 2f * sphere.radius * Mathf.Max(sc.x, Mathf.Max(sc.y, sc.z));
                size = new Vector3(d, d, d);
                return "sphere";
            }
            if (col is BoxCollider box)
            {
                size = Vector3.Scale(box.size, sc);
                return "box";
            }
            if (col is CapsuleCollider cap)
            {
                float d = 2f * cap.radius;
                var v = new Vector3(d, d, d);
                v[Mathf.Clamp(cap.direction, 0, 2)] = Mathf.Max(cap.height, d);
                size = Vector3.Scale(v, sc);
                return "capsule";
            }
            if (col is CharacterController cc)
            {
                float d = 2f * cc.radius;
                size = Vector3.Scale(new Vector3(d, Mathf.Max(cc.height, d), d), sc);
                return "capsule";
            }
            if (col is WheelCollider wheel)
            {
                float d = 2f * wheel.radius;
                size = new Vector3(d, d, d);
                return "wheel";
            }
            if (col is TerrainCollider)
            {
                size = col.bounds.size;
                return "terrain";
            }
            if (col is MeshCollider mesh && mesh.sharedMesh != null)
            {
                size = Vector3.Scale(mesh.sharedMesh.bounds.size, sc);
                return "mesh";
            }
            size = col.bounds.size;
            return "mesh";
        }

        static string Clean(string s)
        {
            return (s ?? "").Replace(" (Instance)", "").Replace("(Clone)", "").Trim();
        }

        static string F(float v)
        {
            if (float.IsNaN(v) || float.IsInfinity(v)) return "0";
            return v.ToString("0.#####", CultureInfo.InvariantCulture);
        }

        static string Quote(string s)
        {
            var sb = new StringBuilder(s.Length + 2);
            sb.Append('"');
            foreach (char ch in s)
            {
                if (ch == '"') sb.Append("\\\"");
                else if (ch == '\\') sb.Append("\\\\");
                else if (ch < ' ') sb.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                else sb.Append(ch);
            }
            sb.Append('"');
            return sb.ToString();
        }
    }
}
