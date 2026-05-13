---
title: "Pathology (병리)"
layout: archive
permalink: /categories/pathology/
author_profile: true
sidebar_main: true
---

Pathology foundation models and CBM-style interpretability work that operates directly on histology / radiology images.

{% assign cat_posts = site.categories["Pathology"] %}
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
