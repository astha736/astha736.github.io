---
layout: page
title: Amphibious Path Planning
description: How can a salamander-like robot plan and follow paths across water and land while respecting the constraints of undulatory, body-driven locomotion?
img: assets/img/ProjectArt_APP.png
importance: 3
category: Doctoral Research
related_publications: false
header:
  image: "assets/img/ProjectArt_APP.png"
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

This project studies path planning and guidance for amphibious, salamander-like robots that move using body undulations and limb coordination. Unlike wheeled or point-mass robots, these systems cannot instantaneously turn or follow arbitrary geometric paths; their motion is constrained by body dynamics, gait structure, terrain, and the interaction between the controller and the environment.

The broader question is how high-level navigation goals can be translated into physically feasible low-level locomotion strategies. In amphibious settings, the robot must reason across water and land, handle changes in medium and terrain, and follow paths while preserving stable and efficient locomotor behavior.

## Methods and Tools

The work combines high-level path planning, waypoint and heading-based guidance, bio-inspired locomotion control, and physics-based simulation of salamander-like robots. The project explores planning and control strategies such as tree-based search, Dubins-style paths, guiding vector fields, and heading regulation, while studying how these methods interact with the constraints of undulatory locomotion and amphibious movement.
