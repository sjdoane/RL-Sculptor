# Meeting with Lokesh — Analysis & Future Directions

*Produced by a multi-agent pass: 4 agents independently read the (rough) transcript and extracted Lokesh's takeaways + recommended directions; a compiler merged them into the master list below (agreement noted, e.g. 4/4); web-research validators checked each direction against the actual robotics literature; two critique agents assessed the meeting.*

---

## Master list of future directions Lokesh recommended

*(ordered by emphasis; each has a research-validation tag)*

### 1. Add explicit priors — above all a **motion generator** — instead of leaning on a VLM to write reward functions  `(4/4)`
Put a **kinematic motion generator** at the front (text / end-effector / waypoint conditioned) that produces a "good kinematic guess" of the behavior; make the RL objective a **tracking reward** against that reference. Demote the LLM to conditioning the generator, high-level mode logic, and at most small residual reward *terms*.
- **Research: active frontier — this is the field's dominant paradigm.** Real work: DeepMimic (Peng 2018), AMP / ASE (Peng 2021–22), PHC, MaskedMimic, CLoSD (closed-loop generate↔track). Lokesh's "kimodo" ≈ NVIDIA-style motion generation (e.g. Kimodo / MaskedMimic).
- **Honest caveat:** his "reward functions are super stupid" is rhetorically strong but *literally false* — Eureka/DrEureka show LLM-written reward **code** genuinely works. The defensible claim: *a single scalar can't carry whole-body, time-extended intent*; a motion prior is a denser signal.

### 2. Replace from-scratch (tabula-rasa MLP) training with **pretrained whole-body tracking controllers** used as a feasibility filter  `(4/4)`
Use an existing "mimicker" tracker to gate generated motions: if the frozen tracker can track a candidate motion → it's feasible, keep it; else discard/tweak. Distill a clean dataset, then a separate downstream training stage → hardware policy.
- **Research: active frontier.** DeepMimic, AMP, PHC (Perpetual Humanoid Control), PULSE, MaskedMimic, OmniH2O/GMT/BeyondMimic (embodiment-matched trackers).
- **Caveats the validators flagged:** (a) tracker feasibility ≠ *hardware* feasibility (it validates against a model, not reality — keep DR in the tracker); (b) use a **force-free** tracker (no cheat forces) or the signal is meaningless; (c) a tracker only knows its training repertoire → it may wrongly reject genuinely novel skills (partly defeats using an LLM to explore). Budget human-in-the-loop for accepted-novel motions.

### 3. Adopt the lab's structured pipeline: **(human video + task prompt + variations spec) → behavior dataset → rewards → policy**, treating a "task" as a *bundle of behaviors under variation*  `(4/4)`
Less flashy than pure prompting but reliable/scalable. A task = (reference behavior set) × (variations file), not one behavior.
- **Research: pipeline shape is well-established (DeepMimic, AMP/ASE, HumanPlus, H2O/OmniH2O); the *novel, unclaimed* piece is the reward-function stage** — which is exactly your strength and exactly the lab's gap.
- **Highest-value borrowed idea:** make the tracker the **admission filter** between "generate motions" and "author rewards" so you never optimize physically-impossible references. Make "task = bundle under variation" a real data structure (variation-parametric reward templates), and reuse your existing physics-DR / world-DR / RSI as the variation engine.
- **Caveat:** "prompting is naive" is too strong — the strongest systems use LLM/VLM front-ends heavily. The durable distinction is *structured, tracker-grounded, dataset-mediated* prompting vs. one-shot end-to-end prompting.

### 4. Ground your stage-decomposition in the **OGMP "modes" / hybrid-automata** formalism, and treat reward-generation-from-modes as the open research problem  `(4/4)`
Formalize each stage as a "mode" (a bundled behavior) with transition guards; run per-mode reward synthesis.
- **Research: real + directly relevant.** OGMP: *Oracle Guided Multi-mode Policies* (arXiv 2403.04205) + the Preferenced-OGMP loco-manipulation follow-up (2410.01030); the classic formal cousin is **Reward Machines** (Icarte et al.). This is Lokesh's own paper — he explicitly said your stage-decomposition rediscovered its core insight.
- **How to apply:** make "modes" a first-class schema in your env_spec (mode = {reference, per-mode reward terms, success predicate}; transitions = {from, to, guard}); run your Eureka-style reward authoring **per mode**; run your nondegeneracy/metric-trust checks **per mode and per transition**. Framing for a paper: *"LLM-generated Reward Machines / OGMP oracles from video"* — the end-to-end composition nobody has published.

### 5. Make controllers **environment-aware, modular, hierarchical** — stop assuming flat ground  `(4/4)`
Feedback / constraints / external dynamics; the manipulated object is *part of* the control problem ("lifting a box changes the dynamics").
- **Research: active frontier.** RMA (rapid motor adaptation), Lee 2020 & Miki 2022 (perceptive locomotion), the parkour line; loco-manipulation NMPC (payload as dynamics); ASE latent-skill hierarchy for the modular/VLA angle.
- **Best concrete move:** privileged-teacher / student split (RMA pattern) so your DR knobs become an *adaptation signal*, not just noise; make the carried object first-class in both dynamics **and** observation/reward.
- **Caveat:** "a flat controller can only recombine" is overstated — perceptive locomotion already left flat-world behind. Sell the *object-as-dynamics* + *modular hierarchy* threads as frontier; treat env-awareness as table-stakes catch-up.

### 6. Apply the **established, task-matched domain-randomization recipe** rather than rediscovering it  `(4/4)`
His volunteered recipe: randomize action rate; MuJoCo soft joint limits; **mass 0.75–1.5**; encoder/observation bias; **force perturbations as velocity pushes** (not force), sized so imparted momentum ≈ **50% of the behavior's expected momentum**. Match the DR to the task.
- **Research: well-established practice.** legged_gym / "Walk in Minutes", Dynamics Randomization (Peng 2018), Humanoid-Gym, delay randomization. Velocity-pushes, mass 0.75–1.5, encoder bias, action-rate randomization are all real, transfer-proven.
- **Honest caveat:** the specific **"~50% of momentum"** number is his *tacit heuristic* — no paper states it; treat it as a calibratable default, not a law. The value is the *structure*, not the exact constants (they're robot-specific). Longer term: wire DrEureka/ADR/DORAEMON-style auto-DR so ranges calibrate from rollouts. → this maps straight onto the physics-DR work already in your repo; make it *task-typed* and *calibratable*.

### 7. Turn it into a **formal research collaboration** — lab hardware, directed-research credits, ~6-month scope  `(4/4)`
Directed research under Prof Quan Nguyen; hardware offered (a ~$40K robot; a Unitree G1 ~$13K + ~$4K SDK/"realification"); potential lead-author thread OR a piece of the lab's motion-generation project. He offered to send the formal onboarding "white letter."
- **A validator sketched a clean 6-month arc:** SIM (mo 0–2: reproduce/beat hand-tuned tracking on N G1 motions — the direct Eureka comparison) → **VERIFICATION (mo 2–4: make your metric-trust "gauntlet" the paper** — "reward sculpting is only safe if a hard-to-game metric gates every edit"; this is the thing scaling labs *don't* have) → HARDWARE (mo 4–6: ASAP-style residual sim2real on the G1 + an actuator network).
- **Hidden dependency to confirm:** a real motion-capture/teleop path on the G1 (needed for real reference rollouts). Prefer the G1 (locomotion, where your DR/RSI already lives) over the $40K arm.

### 8. Explore **"visual behavior adaptation"** — give blind tracking controllers sight via frozen VLM features (Lokesh's own primary thread; the natural alignment point)  `(4/4)`
Adapt pretrained trackers with RL so they solve tasks with semantic/visual awareness; feed **frozen VLM/foundation-model features** (not from-scratch CNNs) and let the controller pick its representation. His brain analogy: cerebrum = semantic (your knowledge-graph), cerebellum = "light enough to lift", medulla = exact forces.
- **Research: active frontier; the *grounding* is well-supported, the specific "let the controller choose its representation via bypass wiring" is a reasonable-but-untested design hypothesis.** Anchors: R3M/MVP/VC-1 (visual pretraining for control), Visual Whole-Body Control (2403.16967), **VideoMimic** (2505.03729, environment-conditioned humanoid control from video), ASE/CALM (high-level over frozen low-level).
- **Sharpest Sam-shaped contribution:** turn VLM semantics ("fragile screen") into a **modulated objective** (compliance / force budget) — a principled answer to "you can't hand-write a force schedule per object." Run the honest baseline first (frozen VC-1 vs from-scratch CNN vs ground-truth state) — "from-scratch is so 2010" isn't universally true.

### 9. Build a **purpose-built, model-flexible agent harness** (your own MCP tool library; keep open-source LLMs in play)  `(3/4)`
Move off raw Claude-terminal plumbing; the lab is building its own MCP library so agents interact; keep it model-agnostic to control cost.
- **Research: sound engineering, not a research result.** Precedents: Eureka/DrEureka loops, Code-as-Policies / SayCan / Voyager, LeRobot + Isaac Lab-Arena.
- **Key nuance:** make model-agnosticism a **router** (hard steps → strongest model; log-parsing/orchestration → cheap/open models), *not* cheapest-only — Eureka's ablation shows an open-only stack underperforms on the hard synthesis steps. "Fable makes people poor" is real, but route, don't downgrade wholesale.

### 10. If it stays a hobby: ship a **clean open-source repo and rename it**  `(4/4)`
Polished GitHub repo + post (Twitter > LinkedIn). "RL Sculptor" is weak — pick one spirited word with a clever etymology (he told the Linux = Linus + "Freax" story). Don't let the agent name it (agents default to bland names).
- **Validators strongly agree strategically:** the LLM-writes-reward frontier is *crowding* (Eureka/DrEureka own that headline), so your moat is **UX + taste + the trust loop**, not the algorithm. Make the "watch it cheat → watch it get fixed" reward-reflection loop the headline demo (great shareable content). Open-sourcing under your name *also* protects provenance before the lab absorbs the reward piece.

### 11. Career: get real **technical "trench time"** before (or instead of) pivoting to PM  `(4/4)`
His Steve Jobs point: the best PMs were the best technical ICs first; do genuine IC time before managing.
- **Research validates the *technical* core** (in LLM-driven RL, the human's job compresses to specifying taste = fitness + safety + priors + failure-mode intuition — and that spec is only as good as the low-level experience behind it). The **career claims are opinion**, not evidence — and it's a recruiter with an axe to grind ("PMs have proxies", "disconnect from LinkedIn"). Take the good part; discount the spin.

---

## What went well in the meeting
- **Show, don't tell** — you screen-shared a live demo fast; it let Lokesh reverse-engineer the system and praise it himself.
- **Positioned vs. the literature** — naming Eureka unprompted and articulating the cost tradeoff signaled you know the field.
- **The "graduate student descent" moment landed** — vivid, honest, and it bonded you two.
- **Honest about limits without collapsing** — admitting you don't know G1 sim2real opened the door to his hardware demo and, effectively, the mentorship.
- **Genuine passion read as real** — "robotics is the thing that's interested me most," "working on it almost every day for 3 months." That persistence is exactly what he's screening for.
- **Composure** — you stayed receptive when he reframed the whole thing as a "hobbyist tool" instead of getting defensive.

## What you could improve
- **Stop apologizing for the front end.** You trashed it repeatedly ("ugly", "vibecoded") — he actually *admired* it. Don't self-deprecate the one thing your evaluator likes.
- **Lead with your genuinely novel parts.** The objective trust-metric, the paper knowledge-graph grounding, the research-before-a-stalled-stage loop — you mentioned them in passing and made *him* do the work of naming why they're clever. Name them as deliberate design decisions.
- **Have one crisp ambition sentence.** He asked "what's the intent?" **three times** — that's a values test, not curiosity. "Pretty much for fun / it'd be cool" three times read as *proxy* — the exact word for people he doesn't want. Have a line like *"prompt-to-hardware policy for behaviors no existing dataset covers."*
- **Ask sharp technical questions — you asked none.** He handed you a full DR recipe and twice offered "I can send you papers." Take the papers. Ask about the pretrained trackers, the MCP library, the OGMP paper. For someone being screened on "the knack," this was the single biggest miss.
- **Name your wedge out loud.** He *said* the lab has motion generation but "haven't at all explored reward functions" — which is exactly what you built. Say it: *"your pipeline is missing the reward synthesis I've already built — that's the complementary piece."* You're complementary, not a supplicant.
- **Don't front-load career ambivalence.** Minutes after "definitely not product managers," you volunteered PM + consulting + Apple + a gap year. Lead with robotics as the through-line; cast PM/design as *downstream of* technical depth (his own Jobs point).
- **Leave with a concrete next step.** You closed on "I'll seriously think about it." Ask for the white letter, ask about weekly hours / authorship / whether it's a publication you lead, and propose a follow-up. You put the whole burden of driving it on him.
- **A little more spine.** When he jabbed ("bad influence"), you just conceded. A confident reframe signals the conviction he recruits for.

## The real decision on the table
Lokesh is **recruiting, not chatting** ("that's why I'm investing this time"). The offer: fold RL Sculptor into the lab's end-to-end architecture, ~6 months, directed-research credits (optional), **G1 + hardware**, mentorship, co-authorship — as **either your own lead thread OR a piece of their project** (push for the *lead* track; it's far more valuable, and you hold the missing reward piece). It's unpaid ("knowledge, experience, hardware") and conditional ("if and only if it's an established direction").

**Strategic read:** this isn't technical-vs-PM. By his *own* Jobs argument, a hardware-validated, co-authored robotics result is the highest-leverage thing you could add to *either* a technical career *or* a credible technical-PM one — built on the asset you already have. The real risk isn't wrong-path, it's **time/opportunity cost** (stacking an unpaid heavy commitment onto a final year that may also hold a year-long Apple internship). Recommended moves: **(1)** open-source RL Sculptor now, under your name (pure upside + protects provenance); **(2)** before committing, get concrete on hours / lead-vs-contributor / authorship / IP / realistic publication timeline; **(3)** force the Apple-vs-research calendar conflict into the open and propose a *scoped pilot* ("one behavior to hardware in N weeks") to test the collaboration without betting your whole year.

---
*Caveat on citations: validators were instructed not to fabricate — foundational papers (DeepMimic, AMP, ASE, PHC, RMA, Eureka, DrEureka, OGMP 2403.04205, VideoMimic, R3M/VC-1) are confirmed real; several very recent 2025–26 arXiv IDs were flagged "verify before citing." Speaker attribution and garbled-term reconstruction are inferred from a rough auto-transcript.*
