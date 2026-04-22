---
title: "Study"
layout: archive
permalink: /study/
author_profile: true
classes: wide
---

Foundational deep-dives: bioinformatics (KEGG pathways, genes, compounds, diseases, drugs), evidence theory (Dempster-Shafer), and other technical topics I'm learning as I work through medical-AI problems.

{% assign studies = site.posts | where_exp: "post", "post.path contains '/_posts/Study/'" | sort: "date" | reverse %}

<p style="color:#666;"><em>{{ studies | size }} posts, most recent first.</em></p>

<hr>

{% for post in studies %}
  {% include archive-single.html %}
{% endfor %}
