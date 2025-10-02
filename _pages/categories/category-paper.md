---
title: "Paper (연구 논문)"
layout: archive
permalink: /categories/paper/
author_profile: true
sidebar_main: true
---

{% assign posts = "" | split: "" %}
{% assign subcategories = site.data.categories.paper.children %}

{% for sub in subcategories %}
  {% assign slug = sub[0] %}
  {% if site.categories[slug] %}
    {% assign posts = posts | concat: site.categories[slug] %}
  {% endif %}
{% endfor %}

{% assign posts = posts | sort: 'date' | reverse %}

{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}
