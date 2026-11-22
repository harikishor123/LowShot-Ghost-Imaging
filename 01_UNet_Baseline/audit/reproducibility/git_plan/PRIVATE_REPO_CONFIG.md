# Proposed private GitHub repository configuration

> **Update 2026-11-05:** the authors decided to make the GitHub repository **public**. The permission-pending exclusions stay in force, and a further 16 files are kept out of the public repository (`07_Reproducibility_and_Audit/EXCLUDED_FROM_PUBLIC_RELEASE.md`). The first commit contains 3,066 files.

**Status: proposal only.** No repository has been created; nothing has been pushed. Items marked **[you]** need your decision. Nothing here assumes names, accounts or licences.

## 1 · Repository

| Setting | Proposed value | Notes |
|---|---|---|
| Owner | **[you]**: a personal account of one team member, or an organisation (for example a lab or course organisation) | ownership decides who can later make it public |
| Name | **[you]**; suggestion: `mtp-ghost-imaging-reconstruction` | avoid personal names in the repository name |
| Visibility | **Private** | stays private for the whole review |
| Description | "Simulated differential ghost imaging with U-Net reconstruction: measurement-shot reduction study (MTP)" | no claims beyond the thesis |
| Initialise with README / licence / .gitignore on GitHub | **No** | the local files are pushed instead; a GitHub-generated licence would apply one automatically |
| Default branch | `main` | |
| Topics (optional) | `ghost-imaging`, `computational-imaging`, `u-net`, `pytorch`, `simulation` | |

## 2 · Access

| Who | Role | Notes |
|---|---|---|
| Owner | Admin | |
| Other team members | **[you]**: Write or Admin | |
| Guide | **[you]**: Read (review only) or Triage | invite by their GitHub account, if they have one |
| Anyone else | none | |

## 3 · Settings to turn off or protect while private

| Setting | Proposed value | Why |
|---|---|---|
| Forking (Settings → General) | disabled | private forks would spread copies of withheld or unreviewed material |
| GitHub Pages | off | Pages from a private repository can be publicly reachable on some plans |
| Wikis, Projects, Discussions | off | not needed |
| Issues | on (optional) | for review comments |
| Actions | disabled, or "allow only local actions" | no CI is needed; avoids accidental runs or secret exposure |
| Releases | none until the Release plan is approved | `release_plan/` is a proposal |
| Branch protection on `main` (if your plan offers it for private repositories) | require a pull request with 1 approval; block force-pushes and deletion | protects the frozen results from accidental rewrites |
| Secret scanning / push protection | on, if available | extra guard; the local scan found no secrets |
| Git LFS | not needed | largest committed file is 25.5 MB |
| Repository size | about 0.39 GB at the first push (3,077 files) | under GitHub's recommended limits; no file over 50 MB |

## 4 · Before making anything public (later, separately)

1. Permission for the onion images, and a decision on the MedMNIST-derived figures (`../EXCLUDED_PENDING_PERMISSION.md`).
2. Licence chosen and `LICENSE` added (`../LICENSING_RECOMMENDATION.md`).
3. Thesis approved by the authors and the guide; institute rules on publishing the thesis checked.
4. Personal data reviewed: names and roll numbers in README and docs, plus local user paths in frozen logs (`/home/ss_students/…`, `C:\Users\vishn\…`). The logs cannot be edited without breaking the freeze manifests.
5. **Material once pushed stays in the Git history.** Anything that must never be public has to stay out of every commit, not just the latest one.
