# Publish Peer Approval Security on GitHub Pages

The website is part of the research repository. GitHub Actions converts saved probe JSON into website data and publishes `docs/`. Deployment does not use Lambda, a GPU, model weights, or a Hugging Face token.

## 1. Know the address and repository requirement

The project URL for this repository is:

**https://subramanyamsahoo.github.io/Peer-Approval-Security/**

`Peer-Approval-Security.github.io` would require a GitHub user or organization with that name and a matching account-site repository. Renaming an ordinary project repository does not grant that hostname. [GitHub site types](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

The earlier upload instructions created a private repository. GitHub Free supports Pages from public repositories; GitHub Pro supports private source repositories too. [GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

A private source repository does not ordinarily make the resulting Pages website private. If using Free, choose whether to make the research repository public or use a separate public site repository. The installer does not change visibility. [Publishing and site visibility](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

## 2. Install and inspect the website files

Use the installer instructions supplied with the website package to add the files to `~/Peer-Approval-Security`. Keep the original experiment folder separate. The old README is preserved as `README_EXPERIMENTS_v02.md`.

The upload should include these paths:

```text
README.md
README_EXPERIMENTS_v02.md
GITHUB_PAGES.md
docs/
tools/export_site_results.py
tools/render_baseline_figures.py
tests/test_export_site_results.py
.github/workflows/pages.yml
```

Check that your checkout points to the intended repository:

```bash
cd ~/Peer-Approval-Security
git remote -v
git status --short
```

If the latest follow-up JSON files are already in this checkout's `results/`, the importer finds them automatically. If they exist only in your original experiment folder, first copy the separate probe output directory into this checkout's `results/`. Do not rename a constructed probe as a baseline run or overwrite the original run.

Build website data and preview:

```bash
python3 tools/export_site_results.py --results-root results --output docs/data/probes.json
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

On a local computer, visit `http://127.0.0.1:8000`. If running this on Lambda, open another terminal on your own computer and forward the port using your existing SSH configuration:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@YOUR_LAMBDA_IP
```

Then visit `http://127.0.0.1:8000` on your computer. Replace `YOUR_LAMBDA_IP` with your actual instance IP and use your usual SSH key if required. Stop the preview with Ctrl+C before continuing in that terminal.

Review the site's evidence labels. Pending means the actual probe JSON has not been imported; it is not a zero effect. The page provides separate runs rather than silently merging probe results.

## 3. Commit and push

From the repository root, stage the website files explicitly:

```bash
git add README.md README_EXPERIMENTS_v02.md GITHUB_PAGES.md docs tools/export_site_results.py tools/render_baseline_figures.py tests/test_export_site_results.py .github/workflows/pages.yml
git diff --cached --stat
git commit -m "Add research website, audited results, and Pages deployment"
git push origin main
```

If new probe output directories have been copied into `results/` but are not committed, add those exact directories in a separate commit before relying on the Actions importer. Its input is the repository checkout in GitHub, not files left only on Lambda. Check `git status --short` to see which files are still local.

No custom deployment secret is needed. The workflow uses GitHub's automatic token and Pages permissions. Do not add `HF_TOKEN`, cloud credentials, private keys, or local environment files to the repository or website.

## 4. Enable GitHub Pages

Open [the repository's Pages settings](https://github.com/SubramanyamSahoo/Peer-Approval-Security/settings/pages).

1. Under **Build and deployment**, set **Source** to **GitHub Actions**.
2. Open the repository's **Actions** tab.
3. Select the Pages workflow, click **Run workflow**, select `main`, and run it.
4. Wait for the build and deployment jobs to succeed. Their deployment summary shows the site URL.

This package uses a custom workflow, so do not select the separate “Deploy from a branch” method. The workflow exports data, uploads the `docs/` artifact, and deploys that artifact. [GitHub custom-workflow instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

Alternatively, after enabling Pages in Settings, trigger and inspect the workflow from Lambda:

```bash
cd ~/Peer-Approval-Security
gh workflow run pages.yml --ref main
gh run list --workflow pages.yml --limit 5
```

To inspect a specific run, copy its run ID from that list:

```bash
gh run view RUN_ID
```

Replace `RUN_ID` with the numeric ID. A workflow must be pushed before it can be dispatched. [GitHub CLI workflow documentation](https://cli.github.com/manual/gh_workflow_run)

## 5. Future updates

Edit the page or commit a new, separate result directory and push to `main`. The workflow will rebuild and redeploy. No experiment is rerun.

```bash
python3 tools/export_site_results.py --results-root results --output docs/data/probes.json
git diff -- docs/data/probes.json
```

Commit the relevant results and website changes after checking them. The importer preserves individual probe runs. It does not select whichever result looks strongest or infer a positive outcome from a missing file.

The baseline JSON is the fixed audited summary of the recorded September 13 run. Adding a follow-up probe does not replace that baseline or retrospectively make its security detector identifiable.

## Troubleshooting

| Symptom | Check |
|---|---|
| Pages asks for a plan upgrade | A private repository on GitHub Free is not eligible. Decide whether to change repository visibility or use a supported plan. |
| `could not find workflow` | Push `.github/workflows/pages.yml` to the default branch first. |
| Permission error during deployment | Confirm Pages source is GitHub Actions and Actions are enabled for the repository. The workflow needs Pages and identity-token write permissions. |
| Workflow reports malformed probe JSON | Read the exporter error and repair the input or keep that incomplete run outside the publication checkout. Do not fabricate summary values. |
| Probe panel still says pending | Confirm the raw probe JSON is committed under `results/`; include its companion summary to display supplied intervals. Check that the latest deployment used that commit. |
| Site returns 404 | Check that deployment succeeded and that the URL includes `/Peer-Approval-Security/`. |
| Page is visible but source links ask for login | The research repository remains private. Pages visibility and repository access are separate. |
| Local preview cannot fetch JSON | Use the Python HTTP server; do not open `index.html` as a `file://` URL. |
| Updated page still looks old | Check the deployed commit, wait for deployment to complete, and refresh the browser. |

The site publishes the contents of `docs/`, including its downloadable JSON. Raw tensors and model checkpoints remain outside the Pages artifact. Hosting the site does not require keeping the Lambda instance running.
