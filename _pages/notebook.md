---
layout: archive
permalink: /notebooks/
title: "Notebooks by tags"
author_profile: true
header:

---


{% include group-by-array collection=site.posts.notebooks field="tags" %}

{% for tag in group_names %}
  {% assign posts.notebooks = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts.notebooks %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
