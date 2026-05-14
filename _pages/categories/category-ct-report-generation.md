---
title: "CT Report Generation"
layout: archive
permalink: /categories/ct-report-generation/
author_profile: true
sidebar_main: true
---

3D CT vision-language models, datasets, and pipelines that produce structured radiology reports from chest / brain / whole-body CT.

{% assign cat_posts = site.categories["CT-Report-Generation"] %}
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
