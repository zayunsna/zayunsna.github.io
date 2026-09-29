"""Weekly Search Console check for zayunsna.github.io (read-only).

Run:  uv run -p 3.12 --with google-api-python-client --with google-auth-oauthlib python scripts/gsc_check.py
      ... scripts/gsc_check.py --no-api     # only check that published posts are live (no login needed)
      ... scripts/gsc_check.py --selftest   # check the flag logic with made-up data

Credentials live OUTSIDE the repo (the repo is public): ~/.config/hk-gsc/client_secret.json and token.json.
Scope is read-only; this script never requests indexing or changes anything in Search Console.
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys
import urllib.request
from zoneinfo import ZoneInfo

SITE_URL = "https://zayunsna.github.io/"
SCOPES = ["https://www.googleapis.com/auth/webmasters.readonly"]
CONFIG_DIR = os.path.expanduser("~/.config/hk-gsc")
CLIENT_FILE = os.path.join(CONFIG_DIR, "client_secret.json")
TOKEN_FILE = os.path.join(CONFIG_DIR, "token.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(REPO, ".gsc")          # gitignored

# Thresholds for "worth a look" flags (small site: keep them low)
LOW_CTR_MIN_IMPRESSIONS = 50
LOW_CTR = 0.02
PAGE_TWO_POSITION = (8.0, 20.0)
PAGE_TWO_MIN_IMPRESSIONS = 20
NOT_CRAWLED_AFTER_DAYS = 7
NO_IMPRESSIONS_AFTER_DAYS = 28

SETUP_GUIDE = f"""Search Console credentials not found: {CLIENT_FILE}
One-time setup (about 10 minutes):
  1. https://console.cloud.google.com/ -> create a project (no billing needed)
  2. APIs & Services -> Library -> enable "Google Search Console API"
  3. APIs & Services -> OAuth consent screen -> External, add your Google account as a test user
  4. Credentials -> Create credentials -> OAuth client ID -> Desktop app -> Download JSON
  5. mkdir -p {CONFIG_DIR} && mv ~/Downloads/client_secret_*.json {CLIENT_FILE}
Then run this script again; a browser window asks you to allow read-only access."""


def today_kst():
    return dt.datetime.now(ZoneInfo("Asia/Seoul")).date()


def load_posts():
    """All posts in the repo as (date, url, title), oldest first."""
    posts = []
    for path in glob.glob(os.path.join(REPO, "*", "_posts", "*.md")):
        category, name = path.split(os.sep)[-3], os.path.basename(path)[:-3]
        date = dt.date.fromisoformat(name[:10])
        m = re.search(r'^title:\s*"?(.*?)"?\s*$', open(path, encoding="utf-8").read(), re.M)
        posts.append((date, f"{SITE_URL}{category}/{name}/", m.group(1) if m else name))
    return sorted(posts)


def check_live(posts, today):
    """Posts dated today or earlier should return 200. A 404 usually means the daily rebuild push was missed."""
    missing = []
    for date, url, title in posts:
        if date > today:
            continue
        try:
            code = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=20).status
        except Exception as e:                     # HTTPError carries .code
            code = getattr(e, "code", str(e))
        if code != 200:
            missing.append((date, url, title, code))
    return missing


# ---------- flag logic (pure functions, covered by --selftest) ----------

def index_flags(inspections, today):
    """inspections: {url: (published_date, indexStatusResult dict)}"""
    flags = []
    for url, (published, r) in inspections.items():
        age = (today - published).days
        if r.get("verdict") != "PASS":
            flags.append((url, f"not indexed: {r.get('coverageState', 'unknown')}"))
        if not r.get("lastCrawlTime") and age > NOT_CRAWLED_AFTER_DAYS:
            flags.append((url, f"never crawled, {age} days after publishing"))
        if r.get("googleCanonical") and r.get("userCanonical") and r["googleCanonical"] != r["userCanonical"]:
            flags.append((url, f"Google chose a different canonical: {r['googleCanonical']}"))
    return flags


def performance_flags(page_rows, published_by_url, today):
    """page_rows: Search Analytics rows grouped by page ({'keys': [url], 'clicks', 'impressions', 'ctr', 'position'})."""
    flags, seen = [], set()
    for row in page_rows:
        url, imp, ctr, pos = row["keys"][0], row["impressions"], row["ctr"], row["position"]
        seen.add(url)
        if imp >= LOW_CTR_MIN_IMPRESSIONS and ctr < LOW_CTR:
            flags.append((url, f"low CTR: {imp:.0f} impressions, CTR {ctr:.1%} -> rewrite title/description"))
        if PAGE_TWO_POSITION[0] <= pos <= PAGE_TWO_POSITION[1] and imp >= PAGE_TWO_MIN_IMPRESSIONS:
            flags.append((url, f"near page one: average position {pos:.1f} -> improve the content"))
    for url, published in published_by_url.items():
        if url not in seen and (today - published).days > NO_IMPRESSIONS_AFTER_DAYS:
            flags.append((url, f"no impressions {(today - published).days} days after publishing -> check indexing"))
    return flags


def selftest():
    today = dt.date(2026, 11, 1)
    ins = {
        "u/ok": (dt.date(2026, 10, 1), {"verdict": "PASS", "lastCrawlTime": "x", "googleCanonical": "u/ok", "userCanonical": "u/ok"}),
        "u/crawled_not_indexed": (dt.date(2026, 10, 1), {"verdict": "NEUTRAL", "coverageState": "Crawled - currently not indexed", "lastCrawlTime": "x"}),
        "u/new": (dt.date(2026, 10, 30), {"verdict": "NEUTRAL", "coverageState": "URL is unknown to Google"}),
        "u/canon": (dt.date(2026, 10, 1), {"verdict": "PASS", "lastCrawlTime": "x", "googleCanonical": "u/other", "userCanonical": "u/canon"}),
    }
    f = index_flags(ins, today)
    assert ("u/crawled_not_indexed", "not indexed: Crawled - currently not indexed") in f
    assert not any(u == "u/new" and "never crawled" in m for u, m in f)          # only 2 days old
    assert any(u == "u/canon" and "canonical" in m for u, m in f)
    assert not any(u == "u/ok" for u, _ in f)
    rows = [{"keys": ["u/a"], "impressions": 400, "ctr": 0.005, "position": 5.0},
            {"keys": ["u/b"], "impressions": 60, "ctr": 0.05, "position": 12.3},
            {"keys": ["u/c"], "impressions": 10, "ctr": 0.0, "position": 30.0}]
    pub = {"u/a": dt.date(2026, 9, 1), "u/b": dt.date(2026, 9, 1), "u/c": dt.date(2026, 9, 1),
           "u/silent": dt.date(2026, 9, 1), "u/fresh": dt.date(2026, 10, 25)}
    p = performance_flags(rows, pub, today)
    assert any(u == "u/a" and "low CTR" in m for u, m in p)
    assert any(u == "u/b" and "near page one" in m for u, m in p)
    assert not any(u == "u/c" for u, _ in p)                                    # too few impressions to judge
    assert any(u == "u/silent" and "no impressions" in m for u, m in p)
    assert not any(u == "u/fresh" for u, _ in p)                                # too new to judge
    print("selftest passed")


# ---------- Search Console API ----------

def service():
    if not os.path.exists(CLIENT_FILE):
        print(SETUP_GUIDE)
        sys.exit(2)
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES) if os.path.exists(TOKEN_FILE) else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file(CLIENT_FILE, SCOPES).run_local_server(port=0)
        with open(os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as fh:
            fh.write(creds.to_json())
    return build("searchconsole", "v1", credentials=creds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="only check that published posts are live")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    today = today_kst()
    posts = load_posts()
    published = [(d, u, t) for d, u, t in posts if d <= today]
    lines = [f"# Search Console check — {today} (KST)", "",
             f"Posts in repo: {len(posts)}, published by today: {len(published)}, scheduled: {len(posts) - len(published)}", ""]

    missing = check_live(posts, today)
    lines += ["## 1. Published posts that are not live", ""]
    lines += [f"- {d} {t} — HTTP {c} ({u})" for d, u, t, c in missing] or ["- none"]
    if any(d == today for d, *_ in missing):
        lines += ["", "**Today's post is not live: run `git commit --allow-empty -m \"rebuild\" && git push`.**"]

    if not args.no_api:
        svc = service()
        sites = [s["siteUrl"] for s in svc.sites().list().execute().get("siteEntry", [])]
        if SITE_URL not in sites:
            print(f"This Google account has no access to {SITE_URL} in Search Console. Properties: {sites}")
            sys.exit(3)

        lines += ["", "## 2. Sitemaps", ""]
        for sm in svc.sitemaps().list(siteUrl=SITE_URL).execute().get("sitemap", []):
            lines.append(f"- {sm['path']}: last downloaded {sm.get('lastDownloaded', 'never')}, "
                         f"errors {sm.get('errors', 0)}, warnings {sm.get('warnings', 0)}, pending {sm.get('isPending', False)}")

        inspections = {}
        for d, u, t in published:
            r = svc.urlInspection().index().inspect(body={"inspectionUrl": u, "siteUrl": SITE_URL}).execute()
            inspections[u] = (d, r.get("inspectionResult", {}).get("indexStatusResult", {}))
        indexed = sum(1 for _, r in inspections.values() if r.get("verdict") == "PASS")
        lines += ["", f"## 3. Indexing ({indexed} of {len(inspections)} published posts indexed)", ""]
        lines += [f"- {u}: {m}" for u, m in index_flags(inspections, today)] or ["- no problems"]

        end = today - dt.timedelta(days=3)                     # final data lags about 2-3 days
        body = {"startDate": str(end - dt.timedelta(days=27)), "endDate": str(end), "dimensions": ["page"], "rowLimit": 1000}
        rows = svc.searchanalytics().query(siteUrl=SITE_URL, body=body).execute().get("rows", [])
        lines += ["", f"## 4. Performance {body['startDate']} to {body['endDate']}", "",
                  f"Clicks {sum(r['clicks'] for r in rows):.0f}, impressions {sum(r['impressions'] for r in rows):.0f}, "
                  f"pages with impressions {len(rows)}", ""]
        lines += [f"- {u}: {m}" for u, m in performance_flags(rows, {u: d for d, u, _ in published}, today)] or ["- nothing to flag"]
        qbody = dict(body, dimensions=["query"], rowLimit=20)
        qrows = svc.searchanalytics().query(siteUrl=SITE_URL, body=qbody).execute().get("rows", [])
        lines += ["", "Top queries:", ""]
        lines += [f"- {r['keys'][0]}: {r['impressions']:.0f} impressions, {r['clicks']:.0f} clicks, "
                  f"position {r['position']:.1f}" for r in qrows] or ["- no queries yet"]

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f"report-{today}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
