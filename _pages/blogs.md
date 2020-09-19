---
layout: archive
permalink: /blog/
title: "My Blog"
author_profile: true
header:
  image: /images/header.PNG
search: true
---

{% include group-by-array collection=site.blogp field="years" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ year | slugify }}" class="archive__subtitle">{{ year }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
