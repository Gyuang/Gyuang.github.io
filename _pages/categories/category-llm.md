---
title: "LLM (Large Language Models)"
layout: archive
permalink: /categories/llm/
author_profile: true
sidebar_main: true
---

Medical multimodal LLMs, tabular LLM feature engineering, and the Concept Bottleneck Model family — anything where a language model is the substrate.

{% assign cat_posts = site.categories["LLM"] %}
{% if cat_posts and cat_posts.size > 0 %}
  {% assign posts = cat_posts | sort: "date" | reverse %}
  <p style="color:#666;"><em>{{ posts | size }} reviews</em></p>
  <hr>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  <p><em>No posts in this category yet.</em></p>
{% endif %}
