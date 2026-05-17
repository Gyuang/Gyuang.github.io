---
title: "LLM Agents"
layout: archive
permalink: /categories/llm-agents/
author_profile: true
sidebar_main: true
---

LLM-as-brain papers — frameworks where a language model acts as the controller that plans, calls external vision / medical / code tools, and synthesizes their outputs. Both medical/bio agents and general multimodal/vision agents.

{% assign cat_posts = site.categories["LLM-Agents"] %}
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
