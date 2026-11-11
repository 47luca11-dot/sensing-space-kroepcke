import json, sys, os, math
here = os.path.dirname(os.path.abspath(__file__))
cams = json.load(open(sys.argv[1], encoding='utf-8'))
sub = sys.argv[2] if len(sys.argv) > 2 else 'tiles'
idx = json.load(open(os.path.join(here, '..', '..', '..', 'web', 'assets', 'city', 'buildings', '_dev', sub, 'index.json'), encoding='utf-8'))
for k, c in cams.items():
    fs = []
    for e in idx['tiles']:
        bb = e['bounds']
        if bb is None: continue
        cx = (bb['min'][0] + bb['max'][0]) / 2; cz = (bb['min'][2] + bb['max'][2]) / 2
        if math.hypot(cx - c['loc'][0], cz - c['loc'][2]) < c.get('radius', 100) + 80:
            fs.append(sub + '/' + e['file'])
            if e.get('windows'): fs.append(sub + '/' + e['windows'])
    fov = 2 * math.degrees(math.atan(12 / c.get('lens', 24))) * 0.5625 / 0.5625
    vfov = 2 * math.degrees(math.atan(18 / c.get('lens', 24) * 9 / 16))
    print(k, 'http://localhost:5177/assets/city/buildings/_dev/viewer.html?tiles=%s&cam=%s&tgt=%s&fov=%.1f&label=%s' % (
        ','.join(fs), ','.join('%.2f' % x for x in c['loc']), ','.join('%.2f' % x for x in c['tgt']), vfov, k))
