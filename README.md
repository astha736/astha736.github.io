# Astha Gupta Website — Local Setup, Docker Notes, and File Map

This repository contains the source code for my personal academic website:

```text
https://astha736.github.io
```

The website is based on the **al-folio** Jekyll theme. I edit the source files locally, preview the site using Docker, and publish by pushing changes to GitHub.


## 1. Mental Model

The website has two main parts:

| Part              | Meaning                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| Source files      | Markdown, YAML, BibTeX, images, layouts, and configuration files that I edit. |
| Generated website | Static HTML/CSS/JS files produced by Jekyll and served by GitHub Pages.       |

The usual workflow is:

```text
Edit source files → Preview locally with Docker → Commit changes → Push to GitHub → GitHub Actions builds site → Website updates online
```

Important branch model:

| Branch     | Meaning                                           | Should I edit it manually? |
| ---------- | ------------------------------------------------- | -------------------------- |
| `master`   | Main source-code branch for the website.          | Yes                        |
| `gh-pages` | Generated deployment branch used by GitHub Pages. | No                         |


| Python world                          | Ruby/Jekyll world   |
| ------------------------------------- | ------------------- |
| `requirements.txt` / `pyproject.toml` | `Gemfile`           |
| Python package                        | Ruby gem            |
| `pip install`                         | `bundle install`    |
| `venv` / environment                  | Bundler environment |
| package lock file                     | `Gemfile.lock`      |


## 2. Docker Setup

Use Docker to avoid installing Ruby, Jekyll, Bundler, and native dependencies directly on macOS.

My Docker Compose file looks like this:

```yaml
# version is not needed anymore - version: "3"
# this file uses prebuilt image from DockerHub
services:
  jekyll:
    image: amirpourmand/al-folio:latest
    ports:
      - 8080:8080
      - 35729:35729
    volumes:
      - .:/srv/jekyll
    environment:
      - JEKYLL_ENV=development
```

### What the Docker Compose file means

| Line / Section                        | Meaning                                                                         | Importance |
| ------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| `services:`                           | Defines the containers needed for this project.                                 | 5/5        |
| `jekyll:`                             | Name of the service/container that runs the website.                            | 4/5        |
| `image: amirpourmand/al-folio:latest` | Uses a prebuilt Docker image containing al-folio/Jekyll dependencies.           | 5/5        |
| `ports: 8080:8080`                    | Maps the container website port to my Mac so I can open `localhost:8080`.       | 5/5        |
| `ports: 35729:35729`                  | Enables live reload when files change.                                          | 3/5        |
| `volumes: .:/srv/jekyll`              | Connects the current repo folder on my Mac to the website folder inside Docker. | 5/5        |
| `JEKYLL_ENV=development`              | Runs the site in development mode.                                              | 4/5        |


## 3. Command Knowledge Table

This table is my running notebook of useful commands.

| Command                     | What it does                                                 | When to use it                                                         |
| --------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `docker --version`          | Checks whether Docker is installed.                          | After installing Docker Desktop.                                       |
| `docker compose version`    | Checks whether Docker Compose is available.                  | Before running the website.                                            |
| `pwd`                       | Prints the current folder path.                              | To confirm I am inside the website repo.                               |
| `ls`                        | Lists files in the current folder.                           | To check that files like `_config.yml` and `docker-compose.yml` exist. |
| `ls docker-compose*`        | Shows available Docker Compose files.                        | To check whether the file is `.yml` or `.yaml`.                        |
| `docker compose config`     | Validates and prints the final Docker Compose configuration. | After editing `docker-compose.yml` or `docker-compose.yaml`.           |
| `docker compose pull`       | Downloads the Docker image defined in the Compose file.      | First setup or when updating the Docker image.                         |
| `docker compose up`         | Starts the local website server.                             | Main command for previewing the website locally.                       |
| `docker compose up --build` | Rebuilds the Docker image before starting.                   | Only needed if using `build: .` or changing the Dockerfile.            |
| `Ctrl + C`                  | Stops the running server in the current terminal.            | When finished previewing.                                              |
| `docker compose down`       | Stops and removes the Compose container/network.             | For cleanup or restarting from a clean state.                          |




## 4. Basic Local Workflow

From inside the repository:

```bash
docker compose pull
docker compose up
```

Then open:

```text
http://localhost:8080
```

When finished:

```text
Ctrl + C
```

Optional cleanup:

```bash
docker compose down
```

---

## 5. Publishing Workflow

After editing and checking the website locally:

```bash
git status
git add .
git commit -m "Update website"
git push origin master
```

Then check GitHub:

```text
Repository → Actions
```

After the deploy action succeeds, open:

```text
https://astha736.github.io
```

---

## 6. File and Folder Organization

This section explains the most important files and folders in the website.

| File / Folder                                 | What it controls                                                                                                                   | Edit often?             | Importance |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ---------- |
| `_config.yml`                                 | Main website configuration: name, URL, social links, theme options, collections, plugins, bibliography settings, enabled features. | Sometimes               | 5/5        |
| `_pages/`                                     | Main pages of the website, such as homepage, bio, publications, CV, teaching, etc.                                                 | Yes                     | 5/5        |
| `_pages/about.md`                             | Usually the homepage or landing page. In this site, it controls the main Research/home content.                                    | Yes                     | 5/5        |
| `_pages/bio.md`                               | Biography page text and profile information.                                                                                       | Yes                     | 5/5        |
| `_pages/publications.md`                      | Publications page layout. Usually pulls data from the BibTeX file.                                                                 | Sometimes               | 4/5        |
| `_bibliography/papers.bib`                    | Publication database in BibTeX format. This controls the publication list.                                                         | Yes, when adding papers | 5/5        |
| `_projects/`                                  | Project cards/pages shown on the website.                                                                                          | Yes                     | 5/5        |
| `_news/`                                      | News or announcements shown on the site.                                                                                           | Sometimes               | 3/5        |
| `_posts/`                                     | Blog posts.                                                                                                                        | Only if blogging        | 3/5        |
| `assets/img/`                                 | Images used in pages, projects, publications, and profile sections.                                                                | Yes                     | 5/5        |
| `assets/`                                     | Static files such as images, CSS, JavaScript, PDFs, and other resources.                                                           | Sometimes               | 4/5        |
| `_data/`                                      | Structured data files used by templates, often YAML files.                                                                         | Rarely                  | 3/5        |
| `_layouts/`                                   | HTML layouts that define page structure.                                                                                           | Rarely                  | 4/5        |
| `_includes/`                                  | Reusable HTML snippets used inside layouts and pages.                                                                              | Rarely                  | 4/5        |
| `_sass/`                                      | Styling source files.                                                                                                              | Rarely                  | 3/5        |
| `_plugins/`                                   | Custom Jekyll plugins.                                                                                                             | Almost never            | 4/5        |
| `.github/workflows/`                          | GitHub Actions deployment workflow. Controls automatic publishing.                                                                 | Rarely                  | 5/5        |
| `docker-compose.yml` or `docker-compose.yaml` | Docker setup for running the site locally.                                                                                         | Rarely                  | 5/5        |
| `Dockerfile`                                  | Instructions for building a custom Docker image. Not needed if using prebuilt image.                                               | Almost never            | 2/5        |
| `Gemfile`                                     | Ruby dependencies for the Jekyll site.                                                                                             | Rarely                  | 4/5        |
| `Gemfile.lock`                                | Exact Ruby dependency versions.                                                                                                    | Almost never manually   | 4/5        |
| `INSTALL.md`                                  | Installation and deployment instructions from al-folio.                                                                            | Read only               | 3/5        |
| `README.md`                                   | My personal notes and project guide.                                                                                               | Yes                     | 5/5        |
| `package.json`                                | Node.js dependencies, mostly for frontend tools.                                                                                   | Rarely                  | 2/5        |
| `purgecss.config.js`                          | CSS cleanup settings for production builds.                                                                                        | Almost never            | 2/5        |
| `robots.txt`                                  | Search-engine crawling instructions.                                                                                               | Rarely                  | 2/5        |

## 7. Most Important Files for Editing Content

If I only want to update my website content, I should usually edit these:

| Goal                                                                     | File / Folder to edit      |
| ------------------------------------------------------------------------ | -------------------------- |
| Change homepage text                                                     | `_pages/about.md`          |
| Change bio                                                               | `_pages/bio.md`            |
| Add or update publication                                                | `_bibliography/papers.bib` |
| Change publications page behavior                                        | `_pages/publications.md`   |
| Add or update projects                                                   | `_projects/`               |
| Add images                                                               | `assets/img/`              |
| Change website title, email, social links, Google Scholar, theme options | `_config.yml`              |

---

## 8. Files to be Careful With

| File / Folder              | Why careful?                                                                  |
| -------------------------- | ----------------------------------------------------------------------------- |
| `_config.yml`              | A small indentation mistake can break the site. YAML is whitespace-sensitive. |
| `.github/workflows/`       | Mistakes can break deployment.                                                |
| `_layouts/`                | Affects the structure of many pages.                                          |
| `_includes/`               | Affects repeated components across the site.                                  |
| `Gemfile` / `Gemfile.lock` | Affects dependency installation and build behavior.                           |
| `docker-compose.yml`       | Affects local Docker setup.                                                   |
| `gh-pages` branch          | This is generated output. Do not edit manually.                               |

---

## 9. YAML Reminder

YAML files are sensitive to spaces and indentation.

Correct:

```yaml
social:
  github: astha736
```

Incorrect:

```yaml
social:
github: astha736
```

In this website, important YAML appears in:

| Place             | Example                        |
| ----------------- | ------------------------------ |
| `_config.yml`     | Site-wide settings             |
| Page front matter | Title, layout, permalink       |
| Project files     | Project title, category, image |
| News files        | Date and announcement content  |

---

## 10. Markdown Reminder

Most content files are Markdown files.

Common Markdown syntax:

| Syntax                    | Meaning         |
| ------------------------- | --------------- |
| `# Title`                 | Main heading    |
| `## Section`              | Section heading |
| `**bold**`                | Bold text       |
| `*italic*`                | Italic text     |
| `[text](url)`             | Link            |
| `![alt text](image_path)` | Image           |
| `- item`                  | Bullet list     |
| `` `code` ``              | Inline code     |

---

## 11. Troubleshooting

| Problem                            | Possible fix                                                       |
| ---------------------------------- | ------------------------------------------------------------------ |
| `docker: command not found`        | Docker Desktop is not installed or not started.                    |
| `docker compose` not found         | Docker Compose plugin is missing or Docker Desktop is not running. |
| Port `8080` already in use         | Change the port mapping or stop the process using port 8080.       |
| Website does not update after edit | Wait a few seconds, refresh browser, or restart Docker Compose.    |
| YAML error                         | Check indentation in `_config.yml` or page front matter.           |
| GitHub site not updated            | Check GitHub Actions tab.                                          |
| Build passes but page looks wrong  | Check `_config.yml`, image paths, and front matter.                |

---

## 12. Personal Learning Notes

Add new commands here as I learn them.

| Date       | Command / Concept        | What I learned                                          |
| ---------- | ------------------------ | ------------------------------------------------------- |
| 2026-06-03 | `docker --version`       | Confirms Docker is installed.                           |
| 2026-06-03 | `docker compose up`      | Starts the local Jekyll website using Docker Compose.   |
| 2026-06-03 | `volumes: .:/srv/jekyll` | My local folder is mounted inside the Docker container. |
| 2026-06-03 | `ports: 8080:8080`       | Makes the website accessible at `localhost:8080`.       |

---

## 13. My Current Recommended Workflow

```bash
cd ~/Code/astha736.github.io

docker compose up
```

Preview:

```text
http://localhost:8080
```

After editing:

```bash
git status
git add .
git commit -m "Update website"
git push origin master
```
