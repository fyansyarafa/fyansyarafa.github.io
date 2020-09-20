---
title: Blogs
layout: collection
permalink: /blog/
collection: blogs
---


{% for post in site.blogp %}
  <h3>
    <a href="{{ post.url }}">{{ post.title }}</a>
  </h3>
  {% for post in site.blogp %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
