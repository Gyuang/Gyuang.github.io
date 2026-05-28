---
title: "Spatial Proteomics"
layout: archive
permalink: /categories/spatial-proteomics/
author_profile: true
sidebar_main: true
---

Computational methods for spatial proteomics — multiplexed imaging platforms (CODEX/Phenocycler, IMC, MIBI, CyCIF, mIF), foundation models, H&E → virtual mIF translation, panel-agnostic cell phenotyping, and clinical-outcome association.

{% assign cat_posts = site.categories["Spatial-Proteomics"] %}
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
