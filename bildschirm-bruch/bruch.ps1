# Gebrochener-Bildschirm-Effekt
# Screenshot + EIN Einschlag: verschobene Glasscherben, ausgebranntes LCD ums Loch,
# leuchtende Risse, flackernde Pixelstreifen und ein Knack-Geraeusch. Maus/Tastatur gesperrt.
# Raus kommt man nur, wenn 6 und 7 GLEICHZEITIG gedrueckt werden (obere Zahlenreihe oder Ziffernblock).
# Notausgang: Strg+Alt+Entf -> Task-Manager -> "Windows PowerShell" beenden.
#
# Aufruf:  bruch.ps1                    -> sofort
#          bruch.ps1 -Verzoegerung 10   -> startet erst nach 10 Sekunden
#          bruch.ps1 -Leise             -> ohne Knack-Geraeusch
#          bruch.ps1 -Vorschau bild.png -> nur Bild speichern, nichts sperren

param(
    [int]$Verzoegerung = 0,
    [switch]$Leise,
    [string]$Vorschau = ""
)

$src = @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Media;
using System.Runtime.InteropServices;
using System.Windows.Forms;

public static class Bruch
{
    [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
    [DllImport("user32.dll")] public static extern short GetAsyncKeyState(int vk);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);

    public class Riss { public List<PointF> P; public float W0, W1; public Riss(List<PointF> p, float w0, float w1) { P = p; W0 = w0; W1 = w1; } }

    static Random R;
    static double Rnd(double a, double b) { return a + R.NextDouble() * (b - a); }
    static float Dist(PointF a, PointF b) { float dx = a.X - b.X, dy = a.Y - b.Y; return (float)Math.Sqrt(dx * dx + dy * dy); }

    // ================= Riss-Geometrie =================

    static List<PointF> Walk(PointF start, double angle, double len, RectangleF lim, double jitter)
    {
        var pts = new List<PointF>();
        pts.Add(start);
        double dir = angle, done = 0;
        float x = start.X, y = start.Y;
        while (done < len)
        {
            double step = Rnd(6, 26);
            dir += Rnd(-jitter, jitter);
            dir = dir * 0.82 + angle * 0.18;
            x += (float)(Math.Cos(dir) * step);
            y += (float)(Math.Sin(dir) * step);
            pts.Add(new PointF(x, y));
            done += step;
            if (!lim.Contains(x, y)) break;
        }
        return pts;
    }

    static PointF? PointAt(List<PointF> pts, PointF c, float r)
    {
        for (int i = 1; i < pts.Count; i++)
            if (Dist(pts[i], c) >= r) return pts[i];
        return null;
    }

    static List<PointF> Jagged(PointF a, PointF b)
    {
        var pts = new List<PointF>();
        pts.Add(a);
        float d = Dist(a, b);
        if (d < 1) { pts.Add(b); return pts; }
        int segs = Math.Max(1, (int)(d / 18));
        float nx = -(b.Y - a.Y) / d, ny = (b.X - a.X) / d;
        for (int i = 1; i < segs; i++)
        {
            float t = (float)i / segs;
            float j = (float)Rnd(-d * 0.07, d * 0.07);
            pts.Add(new PointF(a.X + (b.X - a.X) * t + nx * j, a.Y + (b.Y - a.Y) * t + ny * j));
        }
        pts.Add(b);
        return pts;
    }

    static List<Riss> ErzeugeRisse(RectangleF b, PointF c, float scale, out float kern, List<PointF[]> facetten)
    {
        var risse = new List<Riss>();
        RectangleF lim = RectangleF.Inflate(b, 30, 30);
        float diag = (float)Math.Sqrt(b.Width * b.Width + b.Height * b.Height);

        int n = R.Next(16, 24);
        double a0 = Rnd(0, Math.PI * 2);
        var radials = new List<List<PointF>>();
        for (int i = 0; i < n; i++)
        {
            double ang = a0 + i * Math.PI * 2 / n + Rnd(-0.22, 0.22);
            double len = diag * (R.NextDouble() < 0.6 ? Rnd(0.5, 1.1) : Rnd(0.08, 0.35));
            var p = Walk(c, ang, len, lim, 0.16);
            radials.Add(p);
            risse.Add(new Riss(p, 2.6f, 0.6f));
        }

        var rings = new List<float>();
        float r = (float)Rnd(18, 26) * scale;
        kern = r;
        while (r < 700 * scale) { rings.Add(r); r *= (float)Rnd(1.5, 1.9); }

        // Scherben zwischen zwei Radialrissen und zwei Ringen
        for (int k = 0; k + 1 < rings.Count && k < 6; k++)
            for (int i = 0; i < n; i++)
            {
                int j = (i + 1) % n;
                PointF? p1 = PointAt(radials[i], c, rings[k]), p2 = PointAt(radials[j], c, rings[k]);
                PointF? p3 = PointAt(radials[j], c, rings[k + 1]), p4 = PointAt(radials[i], c, rings[k + 1]);
                if (p1 == null || p2 == null || p3 == null || p4 == null) continue;
                facetten.Add(new PointF[] { p1.Value, p2.Value, p3.Value, p4.Value });
            }

        for (int k = 0; k < rings.Count; k++)
        {
            double prob = Math.Max(0.06, 0.9 - k * 0.2);
            for (int i = 0; i < n; i++)
            {
                if (R.NextDouble() > prob) continue;
                int j = (i + 1) % n;
                PointF? p = PointAt(radials[i], c, rings[k] * (float)Rnd(0.85, 1.15));
                PointF? q = PointAt(radials[j], c, rings[k] * (float)Rnd(0.85, 1.15));
                if (p == null || q == null) continue;
                float w = Math.Max(0.7f, 1.6f - k * 0.25f);
                risse.Add(new Riss(Jagged(p.Value, q.Value), w, w * 0.8f));
            }
        }

        foreach (var rad in radials)
        {
            for (int i = 3; i < rad.Count - 2; i++)
            {
                if (R.NextDouble() > 0.06) continue;
                PointF p = rad[i], q = rad[i + 1];
                double ang = Math.Atan2(q.Y - p.Y, q.X - p.X) + (R.NextDouble() < 0.5 ? -1 : 1) * Rnd(0.35, 0.95);
                risse.Add(new Riss(Walk(p, ang, Rnd(30, 300) * scale, lim, 0.22), 1.2f, 0.4f));
            }
        }
        return risse;
    }

    // Riss in 3 Stuecken zeichnen (verjuengt sich), wenige Pens = schnell
    static void ZeichneRiss(Graphics g, Riss rs, int schattenAlpha)
    {
        var pts = rs.P;
        if (pts.Count < 2) return;
        int teile = Math.Min(3, pts.Count - 1);
        for (int s = 0; s < teile; s++)
        {
            int i0 = (pts.Count - 1) * s / teile, i1 = (pts.Count - 1) * (s + 1) / teile;
            if (i1 <= i0) continue;
            var part = pts.GetRange(i0, i1 - i0 + 1).ToArray();
            float t = (s + 0.5f) / teile;
            float w = rs.W0 + (rs.W1 - rs.W0) * t;
            var sh = new PointF[part.Length];
            for (int i = 0; i < part.Length; i++) sh[i] = new PointF(part[i].X + 1f, part[i].Y + 1f);
            using (var p = new Pen(Color.FromArgb(schattenAlpha, 0, 0, 0), w + 1.4f)) { p.LineJoin = LineJoin.Round; p.StartCap = p.EndCap = LineCap.Round; g.DrawLines(p, sh); }
            using (var p = new Pen(Color.FromArgb(235, 255, 255, 255), w)) { p.LineJoin = LineJoin.Round; p.StartCap = p.EndCap = LineCap.Round; g.DrawLines(p, part); }
        }
    }

    // ================= Ueberbelichtung =================

    static float[] Blur(float[] src, int w, int h, int rad, int passes)
    {
        float[] a = (float[])src.Clone(), t = new float[w * h];
        float norm = 1f / (2 * rad + 1);
        for (int p = 0; p < passes; p++)
        {
            for (int y = 0; y < h; y++)
            {
                int row = y * w; float s = 0;
                for (int x = -rad; x <= rad; x++) s += a[row + Math.Min(w - 1, Math.Max(0, x))];
                for (int x = 0; x < w; x++)
                {
                    t[row + x] = s * norm;
                    s += a[row + Math.Min(w - 1, x + rad + 1)] - a[row + Math.Max(0, x - rad)];
                }
            }
            for (int x = 0; x < w; x++)
            {
                float s = 0;
                for (int y = -rad; y <= rad; y++) s += t[Math.Min(h - 1, Math.Max(0, y)) * w + x];
                for (int y = 0; y < h; y++)
                {
                    a[y * w + x] = s * norm;
                    s += t[Math.Min(h - 1, y + rad + 1) * w + x] - t[Math.Max(0, y - rad) * w + x];
                }
            }
        }
        return a;
    }

    static void Belichte(Bitmap bmp, Rectangle b, PointF c, float scale, List<Riss> risse)
    {
        const int Q = 4;
        int mw = b.Width / Q + 2, mh = b.Height / Q + 2;
        float[] m = new float[mw * mh];
        using (var mb = new Bitmap(mw, mh, PixelFormat.Format32bppArgb))
        {
            using (var g = Graphics.FromImage(mb))
            {
                g.Clear(Color.Black);
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.ScaleTransform(1f / Q, 1f / Q);
                g.TranslateTransform(-b.X, -b.Y);
                foreach (var rs in risse)
                    using (var pen = new Pen(Color.White, (rs.W0 + rs.W1) * 1.2f))
                    {
                        pen.LineJoin = LineJoin.Round;
                        if (rs.P.Count > 1) g.DrawLines(pen, rs.P.ToArray());
                    }
            }
            var md = mb.LockBits(new Rectangle(0, 0, mw, mh), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            byte[] mbytes = new byte[md.Stride * mh];
            Marshal.Copy(md.Scan0, mbytes, 0, mbytes.Length);
            mb.UnlockBits(md);
            for (int y = 0; y < mh; y++)
                for (int x = 0; x < mw; x++)
                    m[y * mw + x] = mbytes[y * md.Stride + x * 4 + 2] / 255f;
        }
        float[] eng = Blur(m, mw, mh, 1, 3);
        float[] weit = Blur(m, mw, mh, 5, 3);

        double[] ph = new double[5];
        for (int i = 0; i < ph.Length; i++) ph[i] = Rnd(0, Math.PI * 2);
        float rc = 34 * scale;       // komplett weisser Kern
        float rf = 85 * scale;       // Abfall der Ueberbelichtung
        float riss0 = 260 * scale;   // wie weit die Risse noch leuchten

        // Belichtung auf 1/4-Aufloesung ausrechnen (schnell), dann nur noch interpolieren
        float[] T = new float[mw * mh];
        for (int my = 0; my < mh; my++)
            for (int mx = 0; mx < mw; mx++)
            {
                float dx = b.X + mx * Q - c.X, dy = b.Y + my * Q - c.Y;
                float d = (float)Math.Sqrt(dx * dx + dy * dy);
                double rad = 0;
                if (d < rc * 2 + rf * 6)
                {
                    double a = Math.Atan2(dy, dx);
                    double noise = 1 + 0.32 * Math.Sin(3 * a + ph[0]) + 0.2 * Math.Sin(5 * a + ph[1])
                                     + 0.12 * Math.Sin(11 * a + ph[2]) + 0.07 * Math.Sin(23 * a + ph[3]);
                    double rcN = rc * (1 + 0.25 * Math.Sin(4 * a + ph[4]));
                    rad = d < rcN ? 5.0 : 5.0 * Math.Exp(-Math.Pow((d - rcN) / (rf * noise), 1.3));
                }
                double fall = Math.Exp(-d / riss0);
                int i = my * mw + mx;
                T[i] = (float)(rad + (eng[i] * 1.8 + weit[i] * 3.0) * (0.08 + 0.92 * fall));
            }

        // Farb-LUTs: leicht blaeulicher Rand wie bei echtem LCD-Ausbluten
        const int L = 1024;
        float[] lE = new float[L], lB = new float[L], lG = new float[L], lR = new float[L];
        for (int i = 0; i < L; i++)
        {
            double e = i / (double)(L - 1);
            lE[i] = (float)e; lB[i] = (float)Math.Pow(e, 0.8); lG[i] = (float)Math.Pow(e, 0.95); lR[i] = (float)Math.Pow(e, 1.05);
        }

        var bd = bmp.LockBits(b, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
        byte[] px = new byte[bd.Stride * b.Height];
        Marshal.Copy(bd.Scan0, px, 0, px.Length);
        for (int y = 0; y < b.Height; y++)
        {
            float v = y / (float)Q;
            int y0 = Math.Min((int)v, mh - 2); float fy = v - y0;
            for (int x = 0; x < b.Width; x++)
            {
                float u = x / (float)Q;
                int x0 = Math.Min((int)u, mw - 2); float fx = u - x0;
                int k = y0 * mw + x0;
                float t0 = T[k] + (T[k + 1] - T[k]) * fx;
                float t1 = T[k + mw] + (T[k + mw + 1] - T[k + mw]) * fx;
                float t = t0 + (t1 - t0) * fy;
                if (t < 0.004f) continue;
                int li = (int)((1 - Math.Exp(-t)) * (L - 1));
                float e = lE[li];
                int i = y * bd.Stride + x * 4;
                // Overlay: weisses Licht (Rand leicht blaeulich) ueber vorhandenes Pixel legen
                float a = px[i + 3] / 255f;
                float oa = e + a * (1 - e);
                if (oa <= 0) continue;
                float gB = 255, gG = 247 + 8 * e, gR = 238 + 17 * e;
                px[i + 0] = (byte)Math.Min(255f, (gB * e + px[i + 0] * a * (1 - e)) / oa);
                px[i + 1] = (byte)Math.Min(255f, (gG * e + px[i + 1] * a * (1 - e)) / oa);
                px[i + 2] = (byte)Math.Min(255f, (gR * e + px[i + 2] * a * (1 - e)) / oa);
                px[i + 3] = (byte)Math.Min(255f, oa * 255);
            }
        }
        Marshal.Copy(px, 0, bd.Scan0, px.Length);
        bmp.UnlockBits(bd);
    }

    static byte Expo(byte v0, float e, float ec)
    {
        float v = v0 * (1 + 1.4f * e) + 12 * e;
        if (v > 255) v = 255;
        v = v + (255 - v) * ec;
        return (byte)(v > 255 ? 255 : v);
    }

    // ================= Scherben verschieben (Lichtbrechung) =================

    static void Scherben(Graphics g, Bitmap src, List<PointF[]> facetten, PointF c, float scale)
    {
        foreach (var f in facetten)
        {
            float d = Dist(f[0], c);
            if (R.NextDouble() > (d < 250 * scale ? 0.75 : 0.35)) continue;
            float maxs = d < 250 * scale ? 5 : 3;
            int dx = (int)Math.Round(Rnd(-maxs, maxs) * scale), dy = (int)Math.Round(Rnd(-maxs, maxs) * scale);
            using (var path = new GraphicsPath())
            {
                path.AddPolygon(f);
                RectangleF bb = path.GetBounds();
                var dst = Rectangle.Round(bb);
                dst.Inflate(2, 2);
                var srcR = new Rectangle(dst.X - dx, dst.Y - dy, dst.Width, dst.Height);
                g.SetClip(path);
                g.DrawImage(src, dst, srcR, GraphicsUnit.Pixel);
                // jede Scherbe faengt das Licht etwas anders
                bool hell = R.NextDouble() < 0.6;
                int alpha = R.Next(0, d < 250 * scale ? 40 : 18);
                using (var br = new SolidBrush(hell ? Color.FromArgb(alpha, 230, 240, 255) : Color.FromArgb(alpha, 0, 0, 0)))
                    g.FillPath(br, path);
                g.ResetClip();
            }
        }
    }

    // ================= LCD-Streifen =================

    static readonly Color[] Farben = {
        Color.FromArgb(255, 255, 0, 255), Color.FromArgb(255, 0, 255, 255),
        Color.FromArgb(255, 0, 255, 60),  Color.FromArgb(255, 255, 255, 255)
    };

    static void Streifen(Graphics g, RectangleF b, PointF c, float scale, int min, int max, int spread)
    {
        int lines = R.Next(min, max);
        for (int i = 0; i < lines; i++)
        {
            float x = c.X + (float)Rnd(-spread, spread) * scale;
            using (var br = new SolidBrush(Color.FromArgb(R.Next(140, 230), Farben[R.Next(Farben.Length)])))
                g.FillRectangle(br, x, b.Top, (float)Rnd(1, 2.5), b.Height);
        }
    }

    // ================= Zusammensetzen =================

    public static Bitmap Glitch;   // zweites Bild fuers Flackern

    public static Bitmap Erzeuge(Rectangle v, int seed)
    {
        R = new Random(seed);
        var bmp = new Bitmap(v.Width, v.Height, PixelFormat.Format32bppArgb);
        Rectangle s = Screen.PrimaryScreen.Bounds;
        var b = new Rectangle(s.X - v.X, s.Y - v.Y, s.Width, s.Height);
        var bf = new RectangleF(b.X, b.Y, b.Width, b.Height);
        float scale = Math.Min(b.Width, b.Height) / 1080f;
        var c = new PointF(b.X + b.Width * (float)Rnd(0.3, 0.7), b.Y + b.Height * (float)Rnd(0.3, 0.65));

        float kern;
        var facetten = new List<PointF[]>();
        List<Riss> risse;
        // durchsichtige Leinwand: nur das Overlay, kein Screenshot
        risse = ErzeugeRisse(bf, c, scale, out kern, facetten);
        using (var g = Graphics.FromImage(bmp))
        {
            g.Clear(Color.Transparent);
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.SetClip(bf);
            // Scherben fangen das Licht unterschiedlich (wie Version 1)
            foreach (var f in facetten)
            {
                float d = Dist(f[0], c);
                bool nah = d < 250 * scale;
                if (!nah && R.NextDouble() > 0.4) continue;
                bool hell = R.NextDouble() < 0.6;
                int alpha = R.Next(4, nah ? 45 : 22);
                using (var br = new SolidBrush(hell ? Color.FromArgb(alpha, 230, 240, 255) : Color.FromArgb(alpha, 0, 0, 0)))
                    g.FillPolygon(br, f);
            }
            Streifen(g, bf, c, scale, 2, 5, 120);
        }

        Belichte(bmp, b, c, scale, risse);

        using (var g = Graphics.FromImage(bmp))
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.SetClip(bf);
            foreach (var rs in risse)
                ZeichneRiss(g, rs, Dist(rs.P[0], c) < 200 * scale ? 60 : 115);
            // zerbroeseltes Glas im Loch
            for (int i = 0; i < 90; i++)
            {
                double a = Rnd(0, Math.PI * 2);
                float d0 = (float)(Math.Pow(R.NextDouble(), 0.7) * kern * 2.0);
                var p = new PointF(c.X + (float)Math.Cos(a) * d0, c.Y + (float)Math.Sin(a) * d0);
                float sz = (float)Rnd(2, 8) * scale;
                var poly = new PointF[R.Next(3, 6)];
                for (int k = 0; k < poly.Length; k++)
                {
                    double pa = k * Math.PI * 2 / poly.Length + Rnd(-0.4, 0.4);
                    poly[k] = new PointF(p.X + (float)Math.Cos(pa) * sz * (float)Rnd(0.4, 1.2), p.Y + (float)Math.Sin(pa) * sz * (float)Rnd(0.4, 1.2));
                }
                int gray = R.Next(140, 220);
                using (var br = new SolidBrush(Color.FromArgb(R.Next(60, 150), gray, gray, Math.Min(255, gray + 10))))
                    g.FillPolygon(br, poly);
            }
            for (int i = 0; i < 60; i++)
            {
                double a = Rnd(0, Math.PI * 2);
                float d0 = (float)Rnd(0, kern * 1.8), d1 = d0 + (float)Rnd(5, 20) * scale;
                double a2 = a + Rnd(-0.7, 0.7);
                var seg = new List<PointF> {
                    new PointF(c.X + (float)Math.Cos(a) * d0, c.Y + (float)Math.Sin(a) * d0),
                    new PointF(c.X + (float)Math.Cos(a2) * d1, c.Y + (float)Math.Sin(a2) * d1) };
                ZeichneRiss(g, new Riss(seg, 1.0f, 0.6f), 70);
            }
        }

        // Flacker-Variante: zusaetzliche Streifen + ein verrutschtes Zeilenband
        Glitch = (Bitmap)bmp.Clone();
        using (var g = Graphics.FromImage(Glitch))
        {
            g.SetClip(bf);
            Streifen(g, bf, c, scale, 3, 8, 260);
            int bandY = (int)(c.Y + Rnd(-300, 300) * scale), bandH = (int)(Rnd(6, 30) * scale);
            using (var br = new SolidBrush(Color.FromArgb(60, 255, 255, 255)))
                g.FillRectangle(br, b.X, bandY, b.Width, bandH);
        }
        return bmp;
    }

    // ================= Knack-Geraeusch =================

    public static byte[] KnackWav()
    {
        const int SR = 44100;
        int n = (int)(SR * 0.7);
        var r = new Random();
        double[] s = new double[n];
        for (int i = 0; i < n; i++)
        {
            double t = i / (double)SR;
            s[i] = (r.NextDouble() * 2 - 1) * (0.95 * Math.Exp(-t / 0.012) + 0.25 * Math.Exp(-t / 0.09));
        }
        // splitterndes Glas: kurze hohe Pings
        for (int k = 0; k < 18; k++)
        {
            int st = (int)(SR * Math.Pow(r.NextDouble(), 1.6) * 0.35);
            double f = 2500 + r.NextDouble() * 6000, amp = 0.08 + r.NextDouble() * 0.25;
            for (int i = 0; i < SR * 0.05 && st + i < n; i++)
            {
                double t = i / (double)SR;
                s[st + i] += amp * Math.Sin(2 * Math.PI * f * t) * Math.Exp(-t / 0.008);
            }
        }
        var ms = new MemoryStream();
        var w = new BinaryWriter(ms);
        w.Write(new char[] { 'R', 'I', 'F', 'F' }); w.Write(36 + n * 2);
        w.Write(new char[] { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' }); w.Write(16);
        w.Write((short)1); w.Write((short)1); w.Write(SR); w.Write(SR * 2); w.Write((short)2); w.Write((short)16);
        w.Write(new char[] { 'd', 'a', 't', 'a' }); w.Write(n * 2);
        for (int i = 0; i < n; i++) w.Write((short)(Math.Max(-1, Math.Min(1, s[i])) * 30000));
        w.Flush();
        return ms.ToArray();
    }

    public static void Start(string vorschau, bool leise)
    {
        SetProcessDPIAware();
        Rectangle v = SystemInformation.VirtualScreen;
        Bitmap bmp = Erzeuge(v, Environment.TickCount);
        if (!string.IsNullOrEmpty(vorschau))
        {
            // Vorschau: Overlay ueber aktuellen Screenshot legen
            foreach (var pair in new[] { new { B = bmp, P = vorschau }, new { B = Glitch, P = Path.ChangeExtension(vorschau, ".glitch.png") } })
            {
                using (var shot = new Bitmap(v.Width, v.Height, PixelFormat.Format32bppArgb))
                {
                    using (var g = Graphics.FromImage(shot)) { g.CopyFromScreen(v.X, v.Y, 0, 0, v.Size); g.DrawImageUnscaled(pair.B, 0, 0); }
                    shot.Save(pair.P, ImageFormat.Png);
                }
            }
            return;
        }
        Application.Run(new BruchForm(bmp, Glitch, v, leise));
    }
}

// Durchsichtiges, durchklickbares Overlay. Laptop bleibt normal benutzbar.
public class BruchForm : Form
{
    [StructLayout(LayoutKind.Sequential)] struct PT { public int x, y; public PT(int a, int b) { x = a; y = b; } }
    [StructLayout(LayoutKind.Sequential)] struct SZ { public int cx, cy; public SZ(int a, int b) { cx = a; cy = b; } }
    [StructLayout(LayoutKind.Sequential, Pack = 1)] struct BLEND { public byte Op, Flags, Alpha, Format; }
    [DllImport("user32.dll")] static extern bool UpdateLayeredWindow(IntPtr h, IntPtr dst, ref PT pos, ref SZ size, IntPtr src, ref PT srcPos, int key, ref BLEND bl, int flags);
    [DllImport("user32.dll")] static extern IntPtr GetDC(IntPtr h);
    [DllImport("user32.dll")] static extern int ReleaseDC(IntPtr h, IntPtr dc);
    [DllImport("gdi32.dll")] static extern IntPtr CreateCompatibleDC(IntPtr dc);
    [DllImport("gdi32.dll")] static extern IntPtr SelectObject(IntPtr dc, IntPtr o);
    [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);

    Timer t;
    int ticks, glitchRest;
    bool glitchAn;
    IntPtr dcNormal, dcGlitch;
    Rectangle v;
    Random r = new Random();
    SoundPlayer knack;

    public BruchForm(Bitmap bmp, Bitmap gl, Rectangle v, bool leise)
    {
        this.v = v;
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        Bounds = v;
        ShowInTaskbar = false;
        IntPtr scr = GetDC(IntPtr.Zero);
        dcNormal = CreateCompatibleDC(scr); SelectObject(dcNormal, bmp.GetHbitmap(Color.FromArgb(0)));
        dcGlitch = CreateCompatibleDC(scr); SelectObject(dcGlitch, gl.GetHbitmap(Color.FromArgb(0)));
        ReleaseDC(IntPtr.Zero, scr);
        bmp.Dispose(); gl.Dispose();
        if (!leise) knack = new SoundPlayer(new MemoryStream(Bruch.KnackWav()));
        Shown += delegate { Zeige(dcNormal); if (knack != null) knack.Play(); };
        t = new Timer();
        t.Interval = 30;
        t.Tick += Tick;
        t.Start();
    }

    protected override bool ShowWithoutActivation { get { return true; } }

    protected override CreateParams CreateParams
    {
        get
        {
            var cp = base.CreateParams;
            // LAYERED | TRANSPARENT (durchklickbar) | TOOLWINDOW (kein Alt+Tab) | NOACTIVATE | TOPMOST
            cp.ExStyle |= 0x80000 | 0x20 | 0x80 | 0x08000000 | 0x8;
            return cp;
        }
    }

    void Zeige(IntPtr dc)
    {
        IntPtr scr = GetDC(IntPtr.Zero);
        var pos = new PT(v.X, v.Y); var size = new SZ(v.Width, v.Height); var src = new PT(0, 0);
        var bl = new BLEND { Op = 0, Flags = 0, Alpha = 255, Format = 1 };
        UpdateLayeredWindow(Handle, scr, ref pos, ref size, dc, ref src, 0, ref bl, 2);
        ReleaseDC(IntPtr.Zero, scr);
    }

    static bool Down(int vk) { return (Bruch.GetAsyncKeyState(vk) & 0x8000) != 0; }

    void Tick(object s, EventArgs e)
    {
        bool sechs = Down(0x36) || Down(0x66);   // 6 oben / Ziffernblock
        bool sieben = Down(0x37) || Down(0x67);  // 7 oben / Ziffernblock
        if (sechs && sieben) { t.Stop(); Application.Exit(); return; }

        // gelegentliches Flackern
        if (glitchRest > 0)
        {
            if (--glitchRest == 0) { glitchAn = false; Zeige(dcNormal); }
            else if (r.NextDouble() < 0.4) { glitchAn = !glitchAn; Zeige(glitchAn ? dcGlitch : dcNormal); }
        }
        else if (r.NextDouble() < 0.012)
        {
            glitchRest = r.Next(2, 9);
            glitchAn = true;
            Zeige(dcGlitch);
        }

        // immer ganz oben bleiben, ohne Fokus zu klauen
        if (++ticks % 15 == 0)
            SetWindowPos(Handle, new IntPtr(-1), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010);
    }
}
'@

# Kompilierten Code zwischenspeichern -> ab dem 2. Start kein Kompilieren mehr
$sha = New-Object Security.Cryptography.SHA1Managed
$hash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($src))) -replace '-', '').Substring(0, 12)
$dll = Join-Path $env:TEMP "bildschirm_bruch_$hash.dll"
if (-not (Test-Path $dll)) {
    Add-Type -TypeDefinition $src -ReferencedAssemblies System.Windows.Forms, System.Drawing -OutputAssembly $dll -OutputType Library
}
Add-Type -Path $dll

if ($Verzoegerung -gt 0) { Start-Sleep -Seconds $Verzoegerung }
[Bruch]::Start($Vorschau, [bool]$Leise)
