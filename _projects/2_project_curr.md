---
layout: page
title: Sim-to-Real Transfer
description: How can bio-inspired controllers help bridge the gap between physics-based simulation and real robotic locomotion?
img: assets/img/ProjectArt_SimToReal.png
importance: 6
category: Doctoral Research
related_publications: false
header:
  image: "assets/img/ProjectArt_SimToReal.png"
status: Finished
---

{% if page.status %}
<span class="badge badge-pill badge-success">{{ page.status }}</span>
{% endif %}

## Overview

This project studies how bio-inspired controllers can support simulation-to-real transfer in robotic locomotion. Simulation provides a powerful environment for testing controllers, perturbations, and design choices, but policies that work in simulation often fail when transferred to hardware because of modeling errors, unmodeled contacts, actuator limits, sensing noise, and environmental variability.

The broader question is whether controllers inspired by biological organization can provide useful structure for transfer. By using rhythmic control, sensory feedback, and morphology-aware strategies, the project investigates how locomotor behaviors can remain stable when moving from idealized simulated conditions to real robotic platforms.

## Methods and Tools

The work used physics-based simulation, bio-inspired control, central pattern generator models, sensory feedback mechanisms, and robotic experiments. Controllers were evaluated across simulated and physical settings to study robustness, transferability, and the role of embodied structure in reducing the gap between model-based predictions and real-world behavior.
