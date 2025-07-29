---
title: "Paper (연구 논문)"
layout: archive
permalink: /categories/paper/
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories.Paper | sort: 'date' | reverse %}
{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}