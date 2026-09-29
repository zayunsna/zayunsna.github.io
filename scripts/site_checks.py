"""Checks to run before every commit that touches the site. Standard library only.

  python3 scripts/site_checks.py links <site_dir>             broken internal links/images, http:// resources
  python3 scripts/site_checks.py future-links [_drafts/x.md=YYYY-MM-DD]... posts linking to posts published later (404 until then)
  python3 scripts/site_checks.py compare <before> <after> [allowed_removed_paths,...]
        sitemap/robots/verification files, post URLs and canonicals unchanged; only the listed pages removed

Build the two sites with scripts/site_build.sh (before your change and after it) for `compare`.
"""
import glob
import html
import os
import re
import sys
from urllib.parse import unquote, urlparse

SITE = "https://zayunsna.github.io"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\]\((?:https://zayunsna\.github\.io)?(/[^)#\s]+/)\)")


def html_pages(root):
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".html"):
                yield os.path.join(d, f)


def links(root):
    def exists(path):
        full = os.path.join(root, unquote(urlparse(path).path).lstrip("/"))
        return os.path.isfile(full) or os.path.isfile(os.path.join(full, "index.html"))
    broken, insecure = {}, set()
    for page in html_pages(root):
        text = open(page, encoding="utf-8", errors="ignore").read()
        for attr, url in re.findall(r'(href|src)="([^"#]+)"', text):
            url = html.unescape(url).replace(SITE, "") or "/"
            if url.startswith("http://") and attr == "src":
                insecure.add(url)
            elif url.startswith("/") and not url.startswith("//") and not exists(url):
                broken.setdefault(url, set()).add(page[len(root):])
    for url, pages in sorted(broken.items()):
        print(f"BROKEN {url}  <- {sorted(pages)[:2]} ({len(pages)} pages)")
    for url in sorted(insecure):
        print(f"HTTP   {url}")
    print(f"broken internal links: {len(broken)}, http resources: {len(insecure)}")
    return 1 if broken or insecure else 0


def future_links(drafts):
    posts = {}
    for f in glob.glob(os.path.join(REPO, "*", "_posts", "*.md")):
        category, name = f.split(os.sep)[-3], os.path.basename(f)[:-3]
        posts[f"/{category}/{name}/"] = (name[:10], f)
    items = list(posts.values()) + [(date, os.path.join(REPO, f)) for f, date in drafts.items()]
    bad = 0
    for date, f in sorted(items):
        for link in LINK.findall(open(f, encoding="utf-8").read()):
            if link in posts and posts[link][0] > date:
                bad += 1
                print(f"{date} {os.path.relpath(f, REPO)} -> {link} (published {posts[link][0]})")
    print(f"future links found: {bad}")
    return 1 if bad else 0


def compare(before, after, allowed):
    def pages(root):
        out = {}
        for page in html_pages(root):
            m = re.search(r'<link rel="canonical" href="([^"]+)"', open(page, encoding="utf-8", errors="ignore").read())
            out[page[len(root):].replace("index.html", "")] = m.group(1) if m else None
        return out
    read = lambda root, f: open(os.path.join(root, f), "rb").read() if os.path.exists(os.path.join(root, f)) else None
    b, a, ok = pages(before), pages(after), True
    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if not cond and detail else ""))
        ok &= cond
    check("sitemap.xml identical (FAIL is expected only if you changed lastmod on purpose)",
          read(before, "sitemap.xml") == read(after, "sitemap.xml"))
    check("robots.txt identical", read(before, "robots.txt") == read(after, "robots.txt"))
    for f in ["google44b8cfdb859086e2.html", "naverd60fc159434ae706f064d74f2d8c2eb5.html", "ads.txt", "feed.xml"]:
        check(f"{f} present", read(after, f) is not None)
    posts = {u: c for u, c in b.items() if re.search(r"/20\d\d-\d\d-\d\d-", u)}
    check("all post URLs still exist", all(u in a for u in posts), [u for u in posts if u not in a][:5])
    check("post canonicals unchanged", all(a.get(u, c) == c for u, c in posts.items()))
    removed = sorted(set(b) - set(a))
    check("only allowed pages removed", set(removed) <= set(allowed), removed)
    print("removed:", removed or "none", "| added:", sorted(set(a) - set(b)) or "none")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "links" and len(args) == 1:
        sys.exit(links(args[0]))
    if cmd == "future-links":
        sys.exit(future_links(dict(a.split("=", 1) for a in args)))
    if cmd == "compare" and len(args) in (2, 3):
        sys.exit(compare(args[0], args[1], args[2].split(",") if len(args) == 3 and args[2] else []))
    print(__doc__)
    sys.exit(2)
