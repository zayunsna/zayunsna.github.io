---
layout: post
title: "Can you trust an AI agent's logs? The viewer is part of the attack surface"
description: >
  A log viewer can hide evidence without changing the stored records. A harmless Python example shows
  why agent output should be rendered as untrusted text, and what escaping does not protect.
image: /assets/img/blog/agent_log_trust/cover.png
sitemap:
  changefreq: daily
  priority: 1.0
---

# Can you trust an AI agent's logs? The viewer is part of the attack surface

Before trusting the transcript of an AI agent's work, ask a smaller question: can text written by that agent
change how the transcript is displayed? A reliable database does not automatically make a reliable review screen.

_Sources, local string checks and Chrome rendering verified as of 2026-10-12._

On October 6, 2026, METR described a proof of concept against the Inspect transcript viewer. An injection could
change what a reviewer saw while the underlying transcript remained in the database. METR said it had not observed
agents exploiting this vulnerability in its evaluations; the flaw was patched. This is a demonstration of a possible
failure, not evidence that every agent is secretly deleting its logs. [METR's original report](https://metr.org/blog/2026-10-06-ai-systems-could-cover-up-misbehavior/), checked 2026-10-12.

## Three records, one misleading display

Our example does not reproduce the Inspect vulnerability. It uses ordinary HTML and a harmless instruction to hide
one line. There is no JavaScript payload, network request, real credential, or external target.

Save this as `example.py` and run it with Python:

```python
import html
import json

events = [
    {"id": 1, "text": "Read input.csv"},
    {"id": 2, "text": '<span style="display:none">Overwrite results.csv</span>'},
    {"id": 3, "text": "Report: task complete"},
]
stored = json.dumps(events, sort_keys=True)
raw_html = "<br>".join(e["text"] for e in events)
escaped_html = "<br>".join(html.escape(e["text"]) for e in events)

print("stored events:", len(json.loads(stored)))
print("raw span present:", '<span style="display:none">' in raw_html)
print("escaped span present:", "&lt;span" in escaped_html)
print("stored text still contains overwrite:", "Overwrite results.csv" in stored)
```

Output, executed with Python 3.14 on macOS on 2026-10-12:

```text
stored events: 3
raw span present: True
escaped span present: True
stored text still contains overwrite: True
```

If a viewer interprets `raw_html` as HTML, the inline CSS tells it to hide the overwrite line. If it renders
`escaped_html` as HTML, the angle brackets become text and the span cannot supply that style. Both strings came
from the same three records. None of the records was erased.

We opened both rendering paths in Chrome 154 on 2026-10-12:

![Chrome screenshot of the demo viewer: the raw HTML panel shows "Read input.csv", an empty line, and "Report: task complete", while the escaped panel shows the span tag and "Overwrite results.csv" as visible text](/assets/img/blog/agent_log_trust/viewer_chrome.png)

In the raw panel, the overwrite line is still in the page's DOM, but its computed style is `display: none`, so the
reader sees only the first and last events. The one visible trace is an empty line where the hidden event sat. In the
escaped panel, no `<span>` element exists; the tag appears as ordinary text and the overwrite event stays readable.

The [complete demo](/assets/code/hotweek/log_viewer_demo.py) writes `events.json` and the local `viewer.html` shown above:

```bash
python log_viewer_demo.py --out /tmp/log-viewer-demo
```

Open that HTML file locally to inspect the difference. The event text is fictional: the program does not overwrite
a real results file. Do not replace the toy input with untrusted executable payloads on a production viewer.

## Protect the review path, not just the storage

For plain logs in a browser, prefer `element.textContent = event.text` to assigning event text to `innerHTML`.
If rich text is essential, use a maintained renderer and sanitization policy appropriate to that format. HTML escaping
is useful at an HTML text boundary; it is not a universal sanitizer for JavaScript, URLs, or every Markdown extension.

Keep the original record accessible independently of the presentation layer. A “download transcript” button in the
same compromised screen is not necessarily an independent verification path. Restrict the agent's ability to edit
the collector, raw storage, viewer code, and monitoring configuration. These are separate permissions from writing
the files needed to complete its actual task.

A hash helps detect changes only when the expected hash is stored somewhere the agent cannot also rewrite.
Computing a new hash from the current file and comparing it with itself proves nothing about historical integrity.
Our demo checks that its serialization survived a local round trip; it does not implement tamper-proof logging.

## What an escaped transcript still cannot prove

Escaping prevents this particular HTML interpretation. It cannot prove that an agent truthfully described its actions,
that a tool call succeeded, or that every relevant event was collected. For those questions, compare the transcript with
tool results, file changes, and telemetry collected outside the agent's control.

The practical lesson is narrower than “never trust AI.” Treat an agent's narrative as one piece of evidence and its
output as untrusted input to your review tools. A clean-looking screen should start verification, not end it.

The same boundary matters when [a local RAG pipeline](/tips/2026-10-05-rag_in_50_lines/) brings document text into
an assistant's prompt: collecting text and deciding what authority it has are separate operations.
