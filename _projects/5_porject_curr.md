---
layout: page
title: Inverse Reinforcement Learning
description: Can demonstrations from bio-inspired controllers be used to design reward functions that remain aligned with expert behavior and generalize beyond the demonstrations?
img: assets/img/ProjectArt_IRL.png
importance: 2
category: Doctoral Research
related_publications: false
header:
  image: "assets/img/ProjectArt_IRL.png"
status: Ongoing
---

{% if page.status %}
<span class="badge badge-pill badge-warning">{{ page.status }}</span>
{% endif %}
{% if page.status == "Ongoing" %}

  <p class="text-muted mt-2">
    This project is ongoing. If it interests you, please feel free to reach out in case you would like to know more, discuss it, or explore ways to extend it.
  </p>
{% endif %}

## Overview

This project studies whether demonstrations from bio-inspired controllers can be used to infer reward functions for robot locomotion. Instead of manually designing rewards only from task-level objectives, the goal is to recover reward structure that reflects expert behavior, including coordination, stability, rhythmicity, and robustness.

The broader question is how embodied agents select one behavior among many possible strategies that satisfy the same objective. In locomotion, several policies may move the robot forward, but differ in energy use, robustness, stability, and transfer. This project uses inverse reinforcement learning to investigate how reward design can encode such preferences.

## Methods and Tools

The work used physics-based simulation of undulatory swimmer robots, bio-inspired controller demonstrations, reinforcement learning, inverse reinforcement learning, and score-driven reward construction. Policies were trained and evaluated under different environmental conditions to study generalization, robustness, and the behavioral strategies induced by different reward formulations.
