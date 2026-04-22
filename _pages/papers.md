---
title: "Papers"
layout: archive
permalink: /papers/
author_profile: true
classes: wide
---

Researcher-focused reviews of work in medical AI, vision-language models, prompt tuning, spatial transcriptomics, and adjacent areas. Every review separates the paper's claims from the evidence that supports them and frames the work through a medical-AI lens.

{% assign papers = site.posts | where_exp: "post", "post.path contains '/_posts/Paper/'" | sort: "date" | reverse %}

<p style="color:#666;"><em>{{ papers | size }} reviews, most recent first.</em></p>

<hr>

{% for post in papers %}
  {% include archive-single.html %}
{% endfor %}
