---
title: "Dataset & Benchmark"
layout: archive
permalink: /categories/dataset/
author_profile: true
sidebar_main: true
---

Datasets, benchmarks, and survey papers that define how the field is measured.

{% assign cat_posts = site.categories["Dataset"] %}
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
