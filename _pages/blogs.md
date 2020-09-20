---
title: Blogs
layout: collection
permalink: /blog/
collection: blogs
---

{% for tag in site.blogp %}
  <h3>{{ tag[0] }}</h3>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}
