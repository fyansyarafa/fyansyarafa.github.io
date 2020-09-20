---
layout: archive
permalink: /blog/
title: "My Blog"
author_profile: true

search: true
---

{% include group-by-array collection=site.blogp field="collections" %}

{% for data in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ data | slugify }}" class="archive__subtitle">{{ tag }}</h2>                      
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}



  {% for post in site.blogp %}

      <a href="{{ post.url }}">{{ post.title }}</a>

    {% include archive-single.html %}
  {% endfor %}
