---
title: "VLM Alignment"
layout: archive
permalink: /categories/vlm-alignment/
author_profile: true
sidebar_main: true
---

How vision-language models are aligned — both at the representation level (CLIP/SigLIP losses, scaling laws, projection design, modality gap) and at the preference level (RLHF / DPO / RLAIF variants for hallucination reduction). Surveys ACL / CVPR / NeurIPS / ICLR work that targets either bucket.

{% assign cat_posts = site.categories["VLM-Alignment"] %}
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
