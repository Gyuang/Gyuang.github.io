---
title: "Multimodal Alignment"
layout: archive
permalink: /categories/multimodal-alignment/
author_profile: true
sidebar_main: true
---

Theoretical and empirical analyses of why contrastive multimodal learning works — and where it breaks. Alignment / uniformity, modality gap, cone effect, dimensional collapse, sigmoid vs softmax losses, and how these all play out in medical adaptations.

{% assign cat_posts = site.categories["Multimodal-Alignment"] %}
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
