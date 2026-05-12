---
title: "Bioinformatics (생물정보학)"
layout: archive
permalink: /categories/bioinformatics/
author_profile: true
sidebar_main: true
---

{% assign cat_posts = site.categories["bioinformatics"] %}
{% if cat_posts %}
  {% assign posts = cat_posts | sort: 'date' | reverse %}
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% else %}
  <p><em>No posts in this category yet.</em></p>
{% endif %}