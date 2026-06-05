---
layout: page
title: Research
permalink: /
# header:
#   image: "/assets/img/Amphibious_Banner.jpg"
imagemagick: true
hide_page_header: true
---

<h2 class="mt-5 mb-3">Research Vision</h2>

My research asks how intelligent behavior emerges in systems that act through a physical body, under real-world constraints, and with objectives that rarely define a single correct solution. In embodied tasks, multiple strategies can achieve the same goal, yet differ in robustness, stability, energy use, adaptability, and transfer.

This question sits at the intersection of robotics, machine learning, control, biomechanics, and computational neuroscience. In my research, I am interested in understanding how different elements of an embodied system interact to enable intelligent behavior that remains reliable beyond controlled laboratory settings.

<h4 class="mt-5 mb-3">Grounding Intelligence in Embodiment</h4>

Intelligent behavior in robots is shaped not only by algorithms, but also by the body, sensors, morphology, contact, dynamics, and the environment. Before information is learned, inferred, or optimized by a model, it is already filtered and structured by the embodied system itself.

I therefore view embodiment as a form of structure: it constrains what information is available, enables certain behaviors, and organizes how control and learning interact with the physical world.

<h4 class="mt-5 mb-3">Locomotion as an Experimental Task</h4>

Locomotion is a demanding setting because behavior cannot be separated from the body. Movement emerges through continuous interaction between control, sensing, morphology, contact, and the environment.

This makes locomotion a useful task for studying physically grounded intelligence. In my work, I use locomotion to investigate how feedback, morphology, and control shape behavior that remains robust under perturbations, damage, and environmental change.

<h4 class="mt-5 mb-3">Investigating and Learning from Biological Robustness</h4>

Biological systems, from simple organisms such as worms to complex animals such as humans, remain remarkably capable under changes in terrain, perturbations, injury, and uncertainty. This robustness emerges from the interaction between neural circuits, sensory feedback, musculoskeletal structure, and the physical environment.

<div class="row justify-content-sm-center mt-3 mb-3">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/guinea_fowl_running_robot_fall.webp" title="Animal robustness and robot locomotion" class="img-fluid rounded z-depth-1" %}
    <p class="caption">
      Running over rough terrain: guinea fowl maintain dynamic stability despite a large unexpected change in substrate height. In contrast, legged robots can still fail under terrain changes, perturbations, and contact uncertainty.
    </p>
  </div>
</div>

During my PhD at the BioRobotics Laboratory, EPFL, I studied these questions by modeling animal locomotion principles in bio-inspired robots. I investigated how sensory feedback, neural control, morphology, and physical interaction contribute to adaptive behavior, and which principles may transfer to engineered physical agents.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/salamander2_robot_NeuroInsp.svg" title="Salamander vs Pleurobot" class="img-fluid" %}
    <p class="caption">
      Components of locomotor control in animals and bio-inspired robots. Descending commands, spinal circuits, sensory feedback, body mechanics, and environmental interaction jointly shape adaptive movement.
    </p>
  </div>
</div>

I explored these questions through [projects]({{ "/projects/" | relative_url }}) on sensory feedback, morphology, simulation-to-real transfer, path planning, reinforcement learning, inverse reinforcement learning, and neuro-inspired modulation.
