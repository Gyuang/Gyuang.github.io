---
title: "Papers"
layout: archive
permalink: /papers/
author_profile: true
classes: wide
---

Researcher-focused reviews of work in medical AI, vision-language models, prompt tuning, spatial transcriptomics, and adjacent areas. Every review separates the paper's claims from the evidence that supports them and frames the work through a medical-AI lens.

{% assign papers = site.categories["Paper"] %}
{% if papers == nil or papers == empty %}
{% assign papers = site.categories.paper %}
{% endif %}

{% if papers and papers.size > 0 %}
{% assign papers = papers | sort: "date" | reverse %}

<p style="color:#666;"><em>{{ papers | size }} reviews total.</em></p>

<h3>Browse by category</h3>

<ul style="columns: 2; -webkit-columns: 2; -moz-columns: 2; list-style: none; padding-left: 0;">
<li><a href="#cat-pathology">Pathology</a> &middot; <span style="color:#888;">{{ site.categories["Pathology"] | size }}</span></li>
<li><a href="#cat-spatial-transcriptomics">Spatial Transcriptomics</a> &middot; <span style="color:#888;">{{ site.categories["Spatial-Transcriptomics"] | size }}</span></li>
<li><a href="#cat-bioinformatics">BioInformatics</a> &middot; <span style="color:#888;">{{ site.categories["BioInformatics"] | size }}</span></li>
<li><a href="#cat-llm">LLM</a> &middot; <span style="color:#888;">{{ site.categories["LLM"] | size }}</span></li>
<li><a href="#cat-multimodal-alignment">Multimodal Alignment</a> &middot; <span style="color:#888;">{{ site.categories["Multimodal-Alignment"] | size }}</span></li>
<li><a href="#cat-vlm-alignment">VLM Alignment</a> &middot; <span style="color:#888;">{{ site.categories["VLM-Alignment"] | size }}</span></li>
<li><a href="#cat-llm-agents">LLM Agents</a> &middot; <span style="color:#888;">{{ site.categories["LLM-Agents"] | size }}</span></li>
<li><a href="#cat-ct-report-generation">CT Report Generation</a> &middot; <span style="color:#888;">{{ site.categories["CT-Report-Generation"] | size }}</span></li>
<li><a href="#cat-generative-models">Generative Models</a> &middot; <span style="color:#888;">{{ site.categories["Generative-Models"] | size }}</span></li>
<li><a href="#cat-dataset">Datasets</a> &middot; <span style="color:#888;">{{ site.categories["Dataset"] | size }}</span></li>
</ul>

<hr>

{% assign cat_keys = "Pathology,Spatial-Transcriptomics,BioInformatics,LLM,Multimodal-Alignment,VLM-Alignment,LLM-Agents,CT-Report-Generation,Generative-Models,Dataset" | split: "," %}
{% assign cat_labels = "Pathology,Spatial Transcriptomics,BioInformatics,LLM,Multimodal Alignment,VLM Alignment,LLM Agents,CT Report Generation,Generative Models,Datasets" | split: "," %}
{% assign cat_slugs = "pathology,spatial-transcriptomics,bioinformatics,llm,multimodal-alignment,vlm-alignment,llm-agents,ct-report-generation,generative-models,dataset" | split: "," %}

{% for key in cat_keys %}
{% assign idx = forloop.index0 %}
{% assign cat_posts = site.categories[key] %}
{% if cat_posts and cat_posts.size > 0 %}
{% assign cat_posts = cat_posts | sort: "date" | reverse %}

<h2 id="cat-{{ cat_slugs[idx] }}">{{ cat_labels[idx] }} <span style="color:#888; font-size: 0.7em;">({{ cat_posts | size }})</span></h2>

{% for post in cat_posts %}{% include archive-single.html %}{% endfor %}

<hr>
{% endif %}
{% endfor %}

<h2 id="all-recent">All reviews — most recent first</h2>

{% for post in papers %}{% include archive-single.html %}{% endfor %}

{% else %}
<p><em>No paper reviews available yet.</em></p>
{% endif %}
