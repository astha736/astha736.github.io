# Astha Gupta Website — Local Setup and Workflow

This repository contains the source code for my personal academic website:

```text
https://astha736.github.io
```

The website is based on the **al-folio** Jekyll theme. I edit the source files locally, preview the site using Docker, check formatting with Prettier, and publish by pushing changes to GitHub.

---

## 1. Mental Model

The website has two main parts:

| Part              | Meaning                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| Source files      | Markdown, YAML, BibTeX, images, layouts, and configuration files that I edit. |
| Generated website | Static HTML/CSS/JS files produced by Jekyll and served by GitHub Pages.       |

Usual workflow:

```text
Edit source files
→ Run Prettier formatting check
→ Preview locally with Docker
→ Commit changes
→ Push to GitHub
→ GitHub Actions builds/deploys site
→ Website updates online
```

Important branch model:

| Branch     | Meaning                                           | Should I edit it manually? |
| ---------- | ------------------------------------------------- | -------------------------- |
| `master`   | Main source-code branch for the website.          | Yes                        |
| `gh-pages` | Generated deployment branch used by GitHub Pages. | No                         |

---

## 2. Tool Worlds

This website uses two main tool ecosystems.

| World           | Files                               | Tool manager | Used for                                 |
| --------------- | ----------------------------------- | ------------ | ---------------------------------------- |
| Ruby/Jekyll     | `Gemfile`, `Gemfile.lock`           | Bundler      | Building and serving the Jekyll website. |
| Node/JavaScript | `package.json`, `package-lock.json` | npm / npx    | Running Prettier and frontend tools.     |

Useful comparison:

| Python world                          | Ruby/Jekyll world   | Node world          |
| ------------------------------------- | ------------------- | ------------------- |
| `requirements.txt` / `pyproject.toml` | `Gemfile`           | `package.json`      |
| Python package                        | Ruby gem            | npm package         |
| `pip install`                         | `bundle install`    | `npm install`       |
| `venv`                                | Bundler environment | `node_modules/`     |
| package lock file                     | `Gemfile.lock`      | `package-lock.json` |

---

## 3. Docker Mental Model

Docker helps me avoid installing Ruby, Jekyll, Bundler, and native dependencies directly on macOS.

| Concept          | Meaning                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| Docker image     | A saved toolbox/template, e.g. `amirpourmand/al-folio:latest`, `node:20`. |
| Docker container | A running copy of an image.                                               |
| Docker service   | A named container setup inside `docker-compose.yml`.                      |
| Volume mount     | Shared folder between my Mac and the container.                           |

Important:

```yaml
volumes:
  - .:/srv/jekyll
```

means:

```text
my current repo folder on Mac = /srv/jekyll inside the container
```

So changes to repo files are permanent on my Mac. Changes inside the temporary container itself are not permanent.

---

## 4. Docker Compose Setup

Current `docker-compose.yml`:

```yaml
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

What it means:

| Line / Section                        | Meaning                                              |
| ------------------------------------- | ---------------------------------------------------- |
| `services:`                           | Defines the containers available for this project.   |
| `jekyll:`                             | Name of the service used to run the website.         |
| `image: amirpourmand/al-folio:latest` | Uses a prebuilt al-folio/Jekyll Docker image.        |
| `ports: 8080:8080`                    | Makes the site available at `http://localhost:8080`. |
| `ports: 35729:35729`                  | Enables live reload.                                 |
| `volumes: .:/srv/jekyll`              | Mounts the local repo into the container.            |
| `JEKYLL_ENV=development`              | Runs the site in development mode.                   |

Do not mix these unless intentionally building a custom image:

```yaml
image: amirpourmand/al-folio:latest
build: .
```

Use either the prebuilt image or a custom `build: .` setup.

---

## 5. Basic Local Preview

From inside the repository:

```bash
cd ~/Code/astha736.github.io
docker compose up
```

Then open:

```text
http://localhost:8080
```

Stop the server:

```text
Ctrl + C
```

Optional cleanup:

```bash
docker compose down
```

---

## 6. Ruby/Jekyll Dependency Flow

Ruby dependencies are controlled by:

```text
Gemfile
Gemfile.lock
```

Mental model:

```text
Gemfile      = what Ruby gems I want
Gemfile.lock = exact gem versions Bundler selected
bundle install = install/update gems and lock versions
```

Example command:

```bash
docker compose run --rm jekyll sh -lc "cd /srv/jekyll && bundle install"
```

Meaning:

| Part                      | Meaning                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| `docker compose run --rm` | Start a temporary container and remove it after the command finishes. |
| `jekyll`                  | Use the `jekyll` service from `docker-compose.yml`.                   |
| `sh -lc "..."`            | Run a shell command inside the container.                             |
| `cd /srv/jekyll`          | Go to the mounted website repo.                                       |
| `bundle install`          | Install Ruby gems from `Gemfile` and update `Gemfile.lock` if needed. |

Important persistence rule:

| Thing                             | Permanent?                                           |
| --------------------------------- | ---------------------------------------------------- |
| Temporary container               | No, because of `--rm`.                               |
| Docker image                      | No, running a container does not rewrite the image.  |
| Repo files such as `Gemfile.lock` | Yes, because the repo is mounted into the container. |

Recent deployment fix:

```ruby
gem "rake"
```

was added to `Gemfile` because the GitHub deploy failed while building a native Ruby gem and could not find the `rake` executable.

After changing `Gemfile`, run:

```bash
docker compose run --rm jekyll sh -lc "cd /srv/jekyll && bundle install"
git diff Gemfile Gemfile.lock
git add Gemfile Gemfile.lock
git commit -m "[UPDATE] Add rake dependency for Jekyll build"
```

---

## 7. Node / Prettier Flow

Prettier checks formatting for Markdown, YAML, Liquid, and other source files.

Node dependencies are controlled by:

```text
package.json
package-lock.json
```

This repo uses:

```json
{
  "devDependencies": {
    "@shopify/prettier-plugin-liquid": "1.4.0",
    "prettier": "3.1.1"
  }
}
```

The Jekyll Docker image may have `node` but not `npm`/`npx`. In that case, use the official temporary Node image:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "node -v && npm -v && npx -v"
```

What it does:

| Part              | Meaning                                        |
| ----------------- | ---------------------------------------------- |
| `docker run --rm` | Run a temporary container and remove it after. |
| `-it`             | Interactive terminal.                          |
| `-v "$PWD":/work` | Mount the current repo into `/work`.           |
| `-w /work`        | Use `/work` as the working directory.          |
| `node:20`         | Use the Node.js 20 image with npm and npx.     |

Install Node dependencies locally inside the mounted repo:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npm ci"
```

Then check formatting:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier . --check"
```

Format only the files reported by GitHub Actions:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier --write <reported files from previous command>"
```

Re-check:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier . --check"
```

If Prettier reports local files such as:

```text
astha736.github.io.code-workspace
docs_REAME/
```

do not commit them accidentally unless they are intentionally part of the repo.

---

## 8. Safe Git Workflow

Before committing:

```bash
git status
git diff --stat
git diff --name-only
```

Avoid using this blindly:

```bash
git add .
```

Safer pattern:

```bash
git add file1 file2 file3
git commit -m "Clear commit message"
git push origin master
```

For formatting fixes:

```bash
git add _config.yml _layouts/page.liquid _pages/about_astha.md _pages/about.md _pages/bio.md _projects/3_project_curr.md Customize.md README.md
git commit -m "[FORMAT] Run Prettier on website files"
git push origin master
```

After pushing, check:

```text
GitHub repo → Actions
```

Then open:

```text
https://astha736.github.io
```

---

## 9. Command Knowledge Table

| Command                                                                                      | What it does                                           | When to use it                                                          |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------- |
| `docker --version`                                                                           | Checks whether Docker is installed.                    | After installing Docker Desktop.                                        |
| `docker compose version`                                                                     | Checks whether Docker Compose is available.            | Before running the website.                                             |
| `pwd`                                                                                        | Prints the current folder path.                        | To confirm I am inside the repo.                                        |
| `ls`                                                                                         | Lists files in the current folder.                     | To check that `_config.yml`, `Gemfile`, and `docker-compose.yml` exist. |
| `docker compose config`                                                                      | Validates and prints the Docker Compose configuration. | After editing `docker-compose.yml`.                                     |
| `docker compose pull`                                                                        | Downloads the Docker image defined in Compose.         | First setup or image update.                                            |
| `docker compose up`                                                                          | Starts the local Jekyll website server.                | Main local preview command.                                             |
| `docker compose down`                                                                        | Stops and removes Compose containers/networks.         | Cleanup or clean restart.                                               |
| `docker compose run --rm jekyll sh -lc "cd /srv/jekyll && bundle install"`                   | Runs Bundler inside a temporary Jekyll container.      | After editing `Gemfile`.                                                |
| `docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npm ci"`                       | Installs Node dependencies from `package-lock.json`.   | Before running Prettier in the Node container.                          |
| `docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier . --check"`       | Checks formatting using Prettier.                      | Before committing or after GitHub formatting failure.                   |
| `docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier --write <files>"` | Formats selected files.                                | When Prettier reports formatting issues.                                |
| `git status`                                                                                 | Shows changed and untracked files.                     | Before staging and committing.                                          |
| `git diff --stat`                                                                            | Shows a compact summary of file changes.               | Before inspecting full diffs.                                           |
| `git diff --name-only`                                                                       | Shows only changed filenames.                          | When full diff is too long.                                             |
| `git add <files>`                                                                            | Stages selected files.                                 | Safer than `git add .`.                                                 |
| `git commit -m "message"`                                                                    | Saves staged changes locally.                          | After checking the diff.                                                |
| `git push origin master`                                                                     | Pushes commits to GitHub.                              | To trigger deployment.                                                  |

---

## 10. File and Folder Map

| File / Folder              | What it controls                                                                                 | Edit often?             |
| -------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------- |
| `_config.yml`              | Main site configuration: name, URL, social links, theme options, plugins, bibliography settings. | Sometimes               |
| `_pages/about.md`          | Main homepage / research landing content.                                                        | Yes                     |
| `_pages/bio.md`            | Biography page.                                                                                  | Yes                     |
| `_pages/publications.md`   | Publications page layout. Usually pulls from BibTeX.                                             | Sometimes               |
| `_bibliography/papers.bib` | Publication database.                                                                            | Yes, when adding papers |
| `_projects/`               | Project cards/pages.                                                                             | Yes                     |
| `_news/`                   | News or announcements.                                                                           | Sometimes               |
| `_posts/`                  | Blog posts.                                                                                      | Yes, for blogging       |
| `assets/img/`              | Images used in pages, projects, publications, and profile sections.                              | Yes                     |
| `_data/`                   | Structured YAML data used by templates.                                                          | Rarely                  |
| `_layouts/`                | HTML/Liquid page layouts.                                                                        | Rarely                  |
| `_includes/`               | Reusable HTML/Liquid snippets.                                                                   | Rarely                  |
| `_sass/`                   | Styling source files.                                                                            | Rarely                  |
| `.github/workflows/`       | GitHub Actions deployment, formatting, and checks.                                               | Carefully               |
| `Gemfile`                  | Ruby/Jekyll dependencies.                                                                        | Rarely                  |
| `Gemfile.lock`             | Exact Ruby dependency versions.                                                                  | Do not edit manually    |
| `package.json`             | Node/Prettier dependencies.                                                                      | Rarely                  |
| `package-lock.json`        | Exact Node dependency versions.                                                                  | Do not edit manually    |
| `docker-compose.yml`       | Local Docker setup.                                                                              | Rarely                  |
| `README.md`                | My project quick-start and notes.                                                                | Yes                     |
| `INSTALL.md` / `FAQ.md`    | Upstream al-folio documentation.                                                                 | Mostly read-only        |
| `gh-pages` branch          | Generated deployed website.                                                                      | Never edit manually     |

---

## 11. Most Important Content Files

| Goal                                                      | File / Folder to edit      |
| --------------------------------------------------------- | -------------------------- |
| Change homepage text                                      | `_pages/about.md`          |
| Change bio                                                | `_pages/bio.md`            |
| Add/update publications                                   | `_bibliography/papers.bib` |
| Add/update projects                                       | `_projects/`               |
| Add images                                                | `assets/img/`              |
| Add blog posts                                            | `_posts/`                  |
| Change title, email, social links, Scholar, theme options | `_config.yml`              |

---

## 12. Files to Be Careful With

| File / Folder                        | Why careful?                                     |
| ------------------------------------ | ------------------------------------------------ |
| `_config.yml`                        | YAML indentation mistakes can break the site.    |
| `.github/workflows/`                 | Mistakes can break deployment or checks.         |
| `_layouts/`                          | Affects structure of many pages.                 |
| `_includes/`                         | Affects repeated components across the site.     |
| `Gemfile` / `Gemfile.lock`           | Affects Ruby dependency installation and deploy. |
| `package.json` / `package-lock.json` | Affects Prettier and Node tooling.               |
| `docker-compose.yml`                 | Affects local Docker setup.                      |
| `gh-pages` branch                    | Generated output. Do not edit manually.          |

---

## 13. YAML Reminder

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

Important YAML appears in:

| Place             | Example                        |
| ----------------- | ------------------------------ |
| `_config.yml`     | Site-wide settings             |
| Page front matter | Title, layout, permalink       |
| Project files     | Project title, category, image |
| News files        | Date and announcement content  |

---

## 14. Markdown Reminder

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

## 15. Troubleshooting

| Problem                                  | Possible fix                                                                              |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| `docker: command not found`              | Docker Desktop is not installed or not started.                                           |
| `docker compose` not found               | Docker Compose plugin is missing or Docker Desktop is not running.                        |
| Port `8080` already in use               | Stop the process using port 8080 or change the port mapping.                              |
| Website does not update after edit       | Refresh browser, wait for reload, or restart Docker Compose.                              |
| YAML error                               | Check indentation in `_config.yml` or page front matter.                                  |
| GitHub site not updated                  | Check GitHub Actions tab.                                                                 |
| Build fails with missing `rake`          | Add `gem "rake"` to `Gemfile`, run `bundle install`, commit `Gemfile` and `Gemfile.lock`. |
| `npm: not found` inside Jekyll container | Use the temporary `node:20` Docker image for Prettier.                                    |
| Prettier cannot find Liquid plugin       | Run `npm ci` so `@shopify/prettier-plugin-liquid` is installed.                           |
| `node_modules/` appears                  | Do not commit it. It should remain ignored.                                               |
| `git diff` opens a long view             | Press `q` to quit. If suspended, run `jobs`, then `fg`, then `q`.                         |
| Local workspace file appears untracked   | Do not commit unless intentionally needed. Consider adding it to `.gitignore`.            |

---

## 16. Current Recommended Workflow

```bash
cd ~/Code/astha736.github.io
```

Preview locally:

```bash
docker compose up
```

Open:

```text
http://localhost:8080
```

Check formatting:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npm ci"
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier . --check"
```

If formatting fails, format only the reported files:

```bash
docker run --rm -it -v "$PWD":/work -w /work node:20 sh -lc "npx prettier --write <files>"
```

Check changes:

```bash
git status
git diff --stat
git diff --name-only
```

Commit selected files:

```bash
git add <files>
git commit -m "Update website"
git push origin master
```

Then check:

```text
GitHub repo → Actions
```

After deploy succeeds:

```text
https://astha736.github.io
```
