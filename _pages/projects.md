---
title: "Projects"
layout: single
permalink: /projects/
author_profile: true
classes: wide
---

내가 1저자 또는 주요 공저자로 참여한 연구. 각 카드를 클릭하면 lambertae.github.io 스타일의 standalone landing page로 이동합니다.

<style>
.project-grid { display: grid; grid-template-columns: 1fr; gap: 1.5em; margin-top: 2em; }
@media (min-width: 720px) { .project-grid { grid-template-columns: 1fr 1fr; } }
.project-card { border: 1px solid #ddd; border-radius: 8px; padding: 1em; transition: box-shadow 0.15s; }
.project-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-color: #999; }
.project-card a.card-link { color: inherit; text-decoration: none; display: block; }
.project-card img { width: 100%; height: 180px; object-fit: cover; border-radius: 4px; margin-bottom: 0.6em; background: #f3f3f3; }
.project-card .card-title { font-weight: 600; font-size: 1.05em; margin: 0.2em 0; }
.project-card .card-venue { color: #777; font-size: 0.85em; font-style: italic; margin-bottom: 0.4em; }
.project-card .card-abstract { color: #444; font-size: 0.9em; line-height: 1.45; }
.project-empty { color: #888; text-align: center; padding: 2em; font-style: italic; }
</style>

{% assign published_projects = site.projects | where: "published", true | sort: "order" %}

{% if published_projects.size > 0 %}
<div class="project-grid">
{% for proj in published_projects %}
<div class="project-card"><a class="card-link" href="{{ proj.url | relative_url }}">
{% if proj.thumbnail %}<img src="{{ proj.thumbnail | relative_url }}" alt="{{ proj.title }}">{% else %}<img src="/assets/images/projects/placeholder.png" alt="placeholder">{% endif %}
<div class="card-title">{{ proj.title }}</div>
<div class="card-venue">{{ proj.venue }}</div>
{% if proj.abstract %}<div class="card-abstract">{{ proj.abstract }}</div>{% endif %}
</a></div>
{% endfor %}
</div>
{% else %}
<p class="project-empty">Coming soon — 논문 정보 정리 중입니다.<br>
새 프로젝트는 <code>_projects/&lt;slug&gt;.md</code> 파일을 만들고 front matter에 <code>published: true</code>를 설정하면 자동으로 카드가 추가되고 <code>/projects/&lt;slug&gt;/</code> URL로 standalone landing page가 생성됩니다.</p>
{% endif %}
