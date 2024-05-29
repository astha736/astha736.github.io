---
layout: page
title: Research
permalink: /
header:
  image: "/assets/img/Amphibious_Banner.jpg"
imagemagick: true
#   caption: "Your caption here"
# profile:
#   align: center
#   image: prof_pic.jpg
#   image_circular: false # crops the image to make it circular
#   more_info: >
#     <p>MED 1 1215 (Bâtiment MED) </p>
#     <p>Station 9 </p>
#     <p>1015 Lausanne </p>

# news: false # includes a list of news items
# selected_papers: false # includes a list of papers marked as "selected={true}"
# social: false # includes social icons at the bottom of the page
---

<!-- ![]() -->




Locomotion in real-world environments is particularly challenging for robots due to the unpredictable and dynamic nature of these settings. Unlike controlled laboratory conditions, real-world terrains vary in texture, slope, and stability, requiring robots to navigate uneven surfaces, obstacles, and changes in elevation. Environmental factors such as weather, debris, and interactions with other entities add further complexity. Stability and reliability in robotic locomotion require robust fault tolerance, as events like lesions, perturbations, and sensor failures are inevitable and significantly impact locomotion. While animals exhibit remarkable robustness to such disruptions, allowing them to recover and continue their tasks, robots must achieve similar resilience, especially in high-stakes scenarios like search and rescue. My research, inspired by the robustness seen in animals, investigates enhancing robot locomotion through sensory-motor integration, Central Pattern Generators (CPGs), and morphological adaptation.

Meanwhile animals exhibit remarkable robustness to such disruptions, allowing them to recover and continue their tasks. Robots must achieve similar resilience, especially in high-stakes scenarios like search and rescue. My research, inspired by the robustness seen in animals, investigates enhancement strategies for robot locomotion through sensory-motor integration, Central Pattern Generators (CPGs), and morphological adaptation.

## Why Are Animals Nailing It While Robots Keep Tripping Up?

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/guinea_fowl_running_robot_fall.webp" title="Animal Nailing and Robot Tripping" class="img-fluid rounded z-depth-1" %}
    <p class="caption">
    Running over rough terrain: guinea fowl maintain dynamic stability despite a large unexpected change in substrate height (Daley, Monica A., et al.)[left]. Agility Robotics's Digit [left], and Boston Dynamics's Atlas [right].
    </p> 
  </div>
</div>

"Why Are Animals Nailing It While Robots Keep Tripping Up?" is a question that both roboticists and biologists are eager to answer. Animals benefit from sophisticated locomotor control, utilizing Central Pattern Generators (CPGs)—neural circuits in the spinal cord that generate rhythmic movement patterns—and higher brain centers that provide adaptive and goal-directed motor control. Their musculoskeletal systems offer dynamic flexibility and strength, enabling seamless adaptation to varying terrains. Moreover, animals possess acute sensory perception, integrating real-time environmental feedback to adjust movements instantly. By understanding and emulating these biological principles, robotics can improve and start to approach the performance levels seen in animals.

<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/salamander2_robot_NeuroInsp.svg" title="Salamander vs Pleurobot" class="img-fluid" %}
  <p class="caption">
    Illustration of different components of the locomotor system and their interaction in animals and bio-inspired robots. Mesencephalic Locomotor Region (MLR) is responsible for controlling locomotion initiation, speed and gait transitions. Reticulospinal (RS) neurons receive input from MLR and are responsible for controlling spinal locomotor circuits (CPG circuits).
  </p> 
  </div>
</div>

To address the challenges in robot locomotion in unpredictable real-world environments, I am working on different projects that delves into various aspects of animal locomotion 


{% assign categorized_projects = site.projects | where: "category", "current" | sort: "importance" %}


<div class="container">
  {% for project in categorized_projects %}
    <div class="row justify-content-sm-center project-item mb-5">
      <!-- {% cycle 'left', 'right' as cycle_var %} -->
        {% if project.importance == 1 or project.importance == 3 %}
          <div class="col-sm-6 mt-3 mt-md-0">
            {% include figure.liquid path=project.img title=project.title class="img-fluid rounded z-depth-1" %}
          </div>
          <div class="col-sm-6 mt-3 mt-md-0">
            <h3>{{ project.title }}</h3>
            <p>{{ project.description }}</p>
          </div>
        {% else %}
          <div class="col-sm-6 mt-3 mt-md-0 order-2 order-md-1">
            <h3>{{ project.title }}</h3>
            <p>{{ project.description }}</p>
          </div>
          <div class="col-sm-6 mt-3 mt-md-0 order-1 order-md-2">
            {% include figure.liquid path=project.img title=project.title class="img-fluid rounded z-depth-1" %}
          </div>
        {% endif %}
      </div>
  {% endfor %}
</div>