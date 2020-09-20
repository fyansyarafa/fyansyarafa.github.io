---
layout: archive
permalink: /blog/
title: "My Blog"
author_profile: true

search: true
---


<ul>
  {% for post in site.blogp %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
    {% include archive-single.html %}
  {% endfor %}
</ul>
