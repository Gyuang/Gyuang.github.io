---
title: "Paper (연구 논문)"
layout: archive
permalink: /categories/paper/
author_profile: true
sidebar_main: true
---

{% assign paper_categories = "Transformer,VLM,Multimodal,Prompt Tuning,RAG,Wsi,Brain" | split: "," %}
{% assign posts = "" | split: "," %}
{% for cat in paper_categories %}
  {% if site.categories[cat] %}
    {% assign posts = posts | concat: site.categories[cat] %}
  {% endif %}
{% endfor %}
{% assign posts = posts | sort: 'date' | reverse %}

{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}