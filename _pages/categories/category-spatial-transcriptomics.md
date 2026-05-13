---
title: "Spatial Transcriptomics (공간전사체)"
layout: archive
permalink: /categories/spatial-transcriptomics/
author_profile: true
sidebar_main: true
---

H&E → spatial gene expression prediction and related cross-modal alignment work (Visium / Xenium / pseudo-bulk).

{% assign cat_posts = site.categories["Spatial-Transcriptomics"] %}
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
