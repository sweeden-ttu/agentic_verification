import re

STOP = set("""a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each either etc few for from further had has have having he her here hers herself him
himself his how however i if in into is it its itself just let like may me might more most much my myself now of off on once one only
or other our ours ourselves out over own per rather same she should so some such than that the their theirs them themselves then
there these they this those through thus to too under until up upon us very via was we were what when where which while who whom why
will with within without would you your yours yourself yourselves here's it's that's there's what's let's i'm you're we're they're
i've you've we've they've i'd you'd we'd they'd i'll you'll we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't
don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't""".split())
KEEP = {"no", "not", "nor", "never", "must", "cannot"}
STOP -= KEEP
WORD = re.compile(r"[a-z][a-z0-9_'\-]*[a-z0-9]|[a-z]")


def words(text):
    return [w.strip("'-") for w in WORD.findall(text.lower())]


def units(text):
    ws = words(text)
    uni = {w for w in ws if (len(w) >= 3 or w in KEEP) and w not in STOP}
    bi = set()
    for a, b in zip(ws, ws[1:]):
        if a in STOP or b in STOP:
            continue
        if len(a) < 2 or len(b) < 2:
            continue
        bi.add(f"{a} {b}")
    return uni | bi


def senses_for(unit, lexicon):
    parts = unit.split()
    out = []
    for sense, ws in lexicon.items():
        if any(p in ws for p in parts):
            out.append(sense)
    return out
