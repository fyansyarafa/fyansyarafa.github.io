---
layout: archive
permalink: /blog/
title: "My Blog"
author_profile: true

search: true
---

{% include group-by-array collection=site.posts.notebooks field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>                      
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
