---
title: Blogs
layout: collection
permalink: /blog/
collection: blogs
---

<ul>
  {% for post in site.blogp %}
    <li>
      <h3>
        <a href="{{ post.url }}">{{ post.title }}</a>
      </h3>

    </li>
  {% endfor %}
</ul>
