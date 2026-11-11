"""Foto | Rendering nebeneinander (gleiche Hoehe) mit Beschriftung.
Aufruf: python -I compare.py <foto> <render> <out.png> [titel_links] [titel_rechts]"""
import sys
from PIL import Image, ImageDraw

a = Image.open(sys.argv[1]).convert('RGB'); b = Image.open(sys.argv[2]).convert('RGB')
H = 900
a = a.resize((int(a.width * H / a.height), H), Image.LANCZOS); b = b.resize((int(b.width * H / b.height), H), Image.LANCZOS)
out = Image.new('RGB', (a.width + b.width + 12, H + 44), (20, 20, 20))
out.paste(a, (0, 44)); out.paste(b, (a.width + 12, 44))
d = ImageDraw.Draw(out)
d.text((10, 14), sys.argv[4] if len(sys.argv) > 4 else 'Foto', fill=(240, 240, 240))
d.text((a.width + 22, 14), sys.argv[5] if len(sys.argv) > 5 else 'Modell', fill=(240, 240, 240))
out.save(sys.argv[3])
print('wrote', sys.argv[3], out.size)
