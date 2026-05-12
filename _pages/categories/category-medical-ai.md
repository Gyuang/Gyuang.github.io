---
title: "Medical AI (의료 AI)"
layout: archive
permalink: /categories/medical-ai/
author_profile: true
sidebar_main: true
---

{% assign cat_posts = site.categories["medical-ai"] %}
{% if cat_posts %}
  {% assign posts = cat_posts | sort: 'date' | reverse %}
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  <p><em>No posts in this category yet.</em></p>
{% endif %}