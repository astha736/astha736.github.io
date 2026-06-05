---
layout: page
title: Deep Reinforcement Learning
description: How can deep reinforcement learning reveal which proprioceptive feedback pathways are useful for robust undulatory swimming?
img: assets/img/ProjectArt_DRL_Architecture.png
importance: 4
category: Doctoral Research
related_publications: true
header:
  image: "assets/img/ProjectArt_DRL_Architecture.png"
status: Finished
---

{% if page.status %}
<span class="badge badge-pill badge-success">{{ page.status }}</span>
{% endif %}

## Overview

This project uses deep reinforcement learning to study how proprioceptive feedback contributes to undulatory swimming. Instead of treating feedback as a fixed biological detail, the project asks which feedback pathways are functionally useful when an embodied agent must learn stable and effective locomotion.

The broader question is how sensory information should be organized across the body. During swimming, feedback can vary spatially, topologically, and functionally: it may depend on where sensors are placed, how they connect to the controller, and what information they provide. This project uses learning as a tool to test which forms of proprioceptive feedback support coordination, robustness, and efficient movement.

## Methods and Tools

The work used physics-based simulation of lamprey-inspired swimming, deep reinforcement learning, proprioceptive feedback models, central pattern generator control, and systematic comparisons of feedback architectures. Learned policies were evaluated to understand how different feedback structures influence locomotor performance, stability, and recovery under perturbations.
