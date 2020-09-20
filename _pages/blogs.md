---
title: Blogs
layout: collection
permalink: /blog/
categories: posts notebooks
---

<hr />

{% for post in site.posts %}
  {% if post.path contains 'notebooks' %}
     {% include archive-single.html %}
  {% endif %}
{% endfor %}   
