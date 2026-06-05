---
layout: page
title: Research
permalink: /
header:
  image: "/assets/img/Amphibious_Banner.jpg"
imagemagick: true
hide_page_header: true
---

<h2 class="mt-5 mb-3">Research Vision</h2>

How does intelligent behavior emerge in systems that act through a physical body, under real-world constraints, and with objectives that rarely define a single correct solution? In embodied tasks, multiple strategies can achieve the same goal, yet differ in robustness, stability, energy use, adaptability, and transfer. Understanding the principles that govern why one behavior emerges over another is central to physically grounded intelligence.

This question can be approached through robotics, machine learning, control, biomechanics, and computational neuroscience. In my research, I am interested in investigating how different elements of an embodied system interact with each other and with the environment to enable intelligent behavior that remains reliable beyond controlled laboratory settings.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/ResearchVision_Art.png" title=" " class="img-fluid" %}
    <p class="caption">
      Physically grounded intelligence as an interaction between control, learning, sensing, morphology, mechanics, computation, task objectives, and environmental dynamics.
    </p>
  </div>
</div>

<h4 class="mt-5 mb-3">Grounding Intelligence in Embodiment</h4>

Intelligent behavior in robots is shaped not only by algorithms, but also by the body, sensors, morphology, contact, dynamics, and the environment. Before information is learned, inferred, or optimized by a model, it is already filtered and structured by the embodied system itself, shaping what becomes relevant and how the physical world is represented for action. Thus, embodiment can be viewed as a form of structure. It constrains what information is available, enables certain behaviors, and organizes how control and learning interact with the physical world.

<h4 class="mt-5 mb-3">Locomotion as an Experimental Task</h4>

Locomotion is a fundamental strategy for survival, shared across much of the animal kingdom. From an embodied task perspective, it is a demanding setting because behavior cannot be separated from the body. Movement emerges through continuous interaction between control, sensing, morphology, contact, and the environment. Moreover, it can be studied across different levels of complexity, from undulatory swimming and crawling to walking, hopping, and other forms of coordinated movement. This makes locomotion a useful task for studying physically grounded intelligence. In my work, I use locomotion to investigate how feedback, morphology, and control shape behavior that remains robust under perturbations, damage, and environmental change.

<h4 class="mt-5 mb-3">Investigating and Learning from Biological Robustness</h4>

Biological systems, from simple organisms such as worms to complex animals such as humans, remain remarkably capable under changes in terrain, perturbations, injury, and uncertainty. This robustness emerges from the interaction between neural circuits, sensory feedback, musculoskeletal structure, and the physical environment.

<div class="row justify-content-sm-center mt-5 mb-3">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/guinea_fowl_running_robot_fall.webp" title="Animal robustness and robot locomotion" class="img-fluid rounded z-depth-1" %}
    <p class="caption">
      Locomotion in animals remains robust under unexpected terrain changes, while legged robots can still be brittle under perturbations and contact uncertainty. Left: guinea fowl experiment from Daley et al.; middle: Agility Robotics' Digit; right: Boston Dynamics' Atlas.
    </p>
  </div>
</div>

During my PhD at the BioRobotics Laboratory, EPFL, I studied these questions by modeling animal locomotion principles in bio-inspired robots. I investigated how sensory feedback, neural control, morphology, and physical interaction contribute to adaptive behavior, and which principles may transfer to engineered physical agents.

<div class="row justify-content-sm-center mt-5 mb-3">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/salamander2_robot_NeuroInsp.svg" title="Salamander vs Pleurobot" class="img-fluid" %}
    <p class="caption">
      Components of locomotor control in animals and bio-inspired robots. Descending commands, spinal circuits, sensory feedback, body mechanics, and environmental interaction jointly shape adaptive movement.
    </p>
  </div>
</div>

For a closer look at how these ideas take shape in specific systems and experiments, see my [doctoral research projects]({{ "/projects/" | relative_url }}).
