---
layout: archive
permalink: /blog/
title: "Blog"
author_profile: true
header:
  image: /images/header.PNG
search: true
---
<ul>
  {% for post in site._posts.notebooks %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
