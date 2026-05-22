---
title: "Generative Models"
layout: archive
permalink: /categories/generative-models/
author_profile: true
sidebar_main: true
---

Generative modeling paradigms — diffusion, flow matching, consistency models, drifting models, GANs, and one-step / few-step samplers. Focus on the underlying math (forward / reverse processes, equilibrium conditions, training objectives) and head-to-head FID/FLOP comparisons.

{% assign cat_posts = site.categories["Generative-Models"] %}
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
