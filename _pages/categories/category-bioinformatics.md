---
title: "Bioinformatics (생물정보학)"
layout: archive
permalink: /categories/bioinformatics/
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories["bioinformatics"] | sort: 'date' | reverse %}

{% for post in posts %}
  {% include archive-single.html %}
{% endfor %}