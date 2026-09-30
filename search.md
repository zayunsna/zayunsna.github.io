---
layout: page
title: Search
description: Search all published notes by title, topic, or summary.
permalink: /search/
noindex: true
sitemap: false
---

<form id="site-search-form" class="site-search" role="search">
  <label for="site-search-input">Search the blog</label>
  <div class="site-search-control">
    <input id="site-search-input" name="q" type="search" autocomplete="off"
           placeholder="Try “RAG”, “pandas”, or “LSTM”" enterkeyhint="search">
    <button type="reset">Clear</button>
  </div>
  <p class="site-search-help">Search runs in your browser. No query is sent to a third party.</p>
</form>

<p id="site-search-status" class="site-search-status" role="status" aria-live="polite"></p>
<ul id="site-search-results" class="search-results"></ul>

<noscript>
  <p>Search requires JavaScript. Browse <a href="/blog/">ML &amp; Engineering</a>,
  <a href="/tips/">AI &amp; Developer Tips</a>, or
  <a href="/ds/">Data Analysis &amp; Statistics</a> instead.</p>
</noscript>

<script type="application/json" id="site-search-data">
[
{% for post in site.posts %}
  {
    "title": {{ post.title | jsonify }},
    "url": {{ post.url | relative_url | jsonify }},
    "description": {{ post.description | default: post.excerpt | strip_html | normalize_whitespace | jsonify }},
    "date": {{ post.date | date: "%Y-%m-%d" | jsonify }},
    "category": {{ post.categories | first | jsonify }}
  }{% unless forloop.last %},{% endunless %}
{% endfor %}
]
</script>
