# PolicyIK — where to improve it further (Goal 2)

A prioritized, evidence-backed roadmap of improvements to PolicyIK, grounded in
what the paper does *not* do and in measurements from this 6-DoF MuJoCo testbed.
Ranked by impact × tractability. One item is already implemented and measured.

## Measured in this repo

### ✅ 1. Smarter deployment protocol (implemented)
`mujoco_policyik/deployment.py` — a **controller-agnostic** retry wrapper:
corrected-goal retries (cancel a repeatable steady-state residual) and
transient-goal retries (escape a stalled configuration). Measured on the UR5e
(`scripts/evaluate_deployment.py`, 150 tasks, ≤4 attempts):

| controller | strict (single) | strict (+deployment) | median pos |
|---|---|---|---|
| DLS | 29.3% | **35.3%** | 54 → 26 mm |
| QP  | 30.0% | **37.3%** | 55 → 16 mm |

On classical IK the lift is modest (E[N]≈3, many failures are fundamental), but
the *same wrapper* is what takes the paper's 93.5%-success policy to 98.9% at
E[N]=1.21. **Next:** make the retry smarter than random transients — choose the
escape configuration from a learned "reset" policy or an explicit nullspace move,
to cut E[N] and rescue the hard cases.

### 2. Joint-limit / redundancy handling (dominant failure mode)
In this testbed **50% of DLS strict failures end with a joint within 2% of its
limit** (the paper reports 71–81% for its 7-DoF arm). This is where classical IK
stalls and where PolicyIK's learned redundancy resolution should win most.
**Improvements:** add an explicit joint-limit-avoidance term to the observation/
reward beyond the current `joint_margin` cost; for redundant arms, expose a
nullspace objective the policy can shape; evaluate specifically on near-limit
goals as a stress split.

## High-impact research directions

### 3. Cross-embodiment generalization (highest impact)
The motivation is VLA transfer across arms, yet every arm needs a full retrain
(7-DoF OpenArm, 6-DoF UR5e here, xArm7 — all separate). Condition one policy on
kinematic parameters (URDF/DH embedding, joint count via padding/masking) so a
single policy serves multiple arms. Directly addresses the paper's own premise.

### 4. Learn the error-dependent gain instead of hand-tuning it
`β(e_p, e_R)` and its thresholds (`g_min, d_lo, d_hi, ori_lo/hi` in `actions.py`)
are hand-set per robot. Make `β` a learned head (or self-tuned), removing a
per-robot tuning burden and likely improving the transport↔precision trade-off.

### 5. Bring the latent-alignment critic + near-goal sampling into the sim path
These are only used in the hardware (FlashSAC) recipe. Large-rotation goals need
a joint-space detour where the error first grows — exactly where a standard PPO
critic is weakest. Porting the LAN critic to the PPO/MJX path may close part of
the remaining single-attempt gap.

## Robustness / scope

### 6. Sim-to-real robustness
The observation assumes clean FK pose feedback. Add domain randomization (PD
gains, link mass, sensor/latency noise) and quantify degradation, so a
sim-trained policy transfers without the separate on-robot FlashSAC run.

### 7. Obstacle / collision awareness
The observation is self-state + goal only. Add workspace-obstacle features for
cluttered pick-and-place (the π0.5 VLA use case the paper targets).

### 8. Beyond position control
It is locked to a joint-position + PD interface. A torque/impedance interface
would help contact-rich tasks where a stiff position controller fights the
environment.

### 9. Reward-weight sensitivity / auto-tuning
Oddly precise weights (e.g. `improvement = 26.831`) suggest heavy tuning. A
sensitivity study — or replacing hand-weighted terms with a simpler objective —
would improve reproducibility across robots. (This repo also found the capped
potential `−min(50, ρ)` creates a constant far-field reward that can trap
exploration — a concrete reward-shaping fix worth upstreaming.)

## Suggested order
1 → 2 (cheap, demonstrable here) → 4 → 5 (precision on the current arm) →
3, 6 (generalization & transfer) → 7, 8 (scope).
