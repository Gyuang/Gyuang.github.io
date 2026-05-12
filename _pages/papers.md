---
title: "Papers"
layout: archive
permalink: /papers/
author_profile: true
classes: wide
---

Researcher-focused reviews of work in medical AI, vision-language models, prompt tuning, spatial transcriptomics, and adjacent areas. Every review separates the paper's claims from the evidence that supports them and frames the work through a medical-AI lens.

{% assign papers = site.categories["Paper"] %}
{% if papers == nil or papers == empty %}
  {% assign papers = site.categories.paper %}
{% endif %}

{% if papers and papers.size > 0 %}
  {% assign papers = papers | sort: "date" | reverse %}
  <p style="color:#666;"><em>{{ papers | size }} reviews, most recent first.</em></p>
  <hr>
  {% for post in papers %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  <p><em>No paper reviews available yet.</em></p>
{% endif %}
