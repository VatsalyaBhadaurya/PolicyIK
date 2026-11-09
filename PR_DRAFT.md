# Draft PR for Tinaliu0123/PolicyIK

A minimal, respectful PR that adds a link to the companion MuJoCo port in the
upstream README. (Gentler alternative: post `UPSTREAM_ISSUE.md` first and only
PR this once the maintainers agree to a link.)

## The change (upstream `README.md`)

Insert a short, clearly-unofficial note right after the one-paragraph project
description (just before the Table of Contents):

```markdown
> **Community reimplementation (unofficial):** a from-scratch MuJoCo port on a
> 6-DoF UR5e — CPU (Stable-Baselines3) and GPU (MJX + Brax), reproducing the
> classical IK baselines — is available at
> [VatsalyaBhadaurya/PolicyIK](https://github.com/VatsalyaBhadaurya/PolicyIK).
```

## PR title

docs: link an unofficial MuJoCo (6-DoF UR5e) companion port

## PR body

Hi — small docs PR adding a link to a companion **MuJoCo port** of PolicyIK on a
6-DoF UR5e (CPU + GPU via MJX/Brax), which may help people who want to try the
ideas without an Isaac Sim / Omniverse setup. It reproduces your DLS/QP baselines
(DLS 51.0% strict / 0.95 mm, matching your 50.4% / 0.94 mm) and re-implements the
Sec. 4.1–4.3 formulation; the learned policy isn't yet trained to your 1 mm/0.2°
precision (compute-bound on a laptop GPU), so it's labeled unofficial / in-progress.

Context and figures are in #<issue-number> (if you'd prefer to keep the README
Isaac-only, totally fine to close this — no worries at all). Thanks for the great work!

— Vatsalya Bhadaurya · www.vatsalyaaa.me

## Commands (after you authenticate)

```bash
# 1) one-time: install + authenticate gh
winget install --id GitHub.cli -e            # (or: sudo apt install gh  in WSL)
gh auth login                                # interactive — you do this once

# 2) fork under a non-clashing name, branch, edit, push, open PR
gh repo fork Tinaliu0123/PolicyIK --fork-name PolicyIK-upstream --clone --default-branch-only
cd PolicyIK-upstream
git checkout -b docs/mujoco-port-link
# apply the README note above
git commit -am "docs: link an unofficial MuJoCo (6-DoF UR5e) companion port"
git push -u origin docs/mujoco-port-link
gh pr create --repo Tinaliu0123/PolicyIK --title "docs: link an unofficial MuJoCo (6-DoF UR5e) companion port" --body-file PR_BODY.txt
```
