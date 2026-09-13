# Website implementation checks

Build date: 2026-09-13.

The website is an update for the existing Peer-Approval-Security research repository. It has not been deployed to the owner's GitHub account in this session.

| Check | Result |
|---|---|
| Probe exporter tests | 11 passed: complete branch validation, nonfinite values, duplicate keys and IDs, separate studies, summary consistency, absent data, and atomic output preservation |
| JavaScript syntax | `node --check docs/app.js` passed |
| Local links and assets | HTML fragment targets and referenced local files found |
| Baseline accounting | Completed stage counts sum to 448; held-out outcome categories sum to 32 in each row |
| Installer preservation | Tested against a temporary copy of v0.2: experiment source hashes unchanged, raw result sentinel unchanged, original README preserved |
| Scientific content review | Cross-checked the headline values, stage statuses, history confound, and constructed-probe interpretation against the dated baseline audit |
| Browser rendering and interaction | Not completed: the cloud browser's URL security policy blocked the local preview; no screenshot or mobile-browser pass is claimed |
| GitHub Pages deployment | Workflow prepared from official GitHub documentation; requires upload and Pages configuration in the owner's repository |

The baseline figures are replotted from the audit-rounded summary estimates. The audit and baseline JSON are included. These plots are not a new experimental run or a reconstruction of unavailable raw history branches.

The default probe data file is explicitly `not_imported`. Actual user probe results were not available in this workspace. The installer and deployment workflow import recorded files from the target repository. Synthetic data are confined to exporter tests; they are not published as empirical results.

Preview on your machine with the commands in the README, and inspect desktop and mobile layouts before treating visual verification as complete.
