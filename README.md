# Algoverse---Chrono-MemBench
A Two-Phase Framework for Tracking and Steering Memory Features in Large Language Models
## 6-Week “Crash-Mode” Plan for **Chrono-MemBench**

> **Principles for compression**
>
> 1. **Overlap everything**: data/weights download, SAE fitting, and dashboard plumbing run in parallel.
> 2. **Skip the 125 M toy model**—go straight to Gemma-2 B; unit-tests cover code sanity.
> 3. **Tighten checkpoints**: save every 200 M tokens (≈ once per day on a 24 GB A100) and prune intermediate ones with `dvc gc`.
> 4. **Scope trims**: only one full ablation grid; LLaVA multimodal moved to “future work” unless vision memory is a hard requirement.

---

### **W-1  (Day 1-7)  —  Foundation & First Tokens**

| Day   | Deliverable                     | Key actions                                                                                                                      |
| ----- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 1 – 2 | **Repo boot-strap**             | Init repo, DVC remote, Git LFS, `.venv` + `pip install`.                                                                         |
| 3     | **Data & base weights in DVC**  | Slim Pile + Gemma-2 B + Llama-3-8 B downloaded & `dvc add`.                                                                      |
| 4     | **Smoke tests pass**            | `pytest -m "unit and not slow"`; Colab notebook clone-and-pull.                                                                  |
| 5 – 7 | **Gemma-2 B training *starts*** | `python -m chrono.train model=gemma-2b ...` — checkpoint 0 at 200 M tokens uploaded; Route-SAE job scheduled on that checkpoint. |

---

### **W-2  —  Joint Route-SAE & Early Metrics**

1. **Run Route-SAE *incrementally***

   * Fit on checkpoints 0 & 1 (0 M, 200 M tokens) with **20 % temporal dropout**.
2. **Integrate SAEBench × 8 metrics** ➜ WandB.
3. **Begin hooks implementation** (Cap-Gauge, Weight-Δ, Mem-Absorption) in a feature branch; test against checkpoint replay.
4. **Grafana stack spun up** (Docker compose) with WebSocket listener.
5. **Checkpoint 2** (\~400 M tokens) produced; live Route-SAE basis updated.

*By end of W-2 you have*: ◦ working SAE basis ◦ first birth-curve sketch ◦ dashboard skeleton.

---

### **W-3  —  Live Telemetry Online**

| Stream A (training loop)                                                             | Stream B (infra)                                                       |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Resume Gemma at 400 M tokens **with Phase B hooks ON** (emit JSON every 10 k steps). | Finalise Triton kernel & WebSocket pipe; verify < 150 ms dash latency. |
| At \~800 M tokens: first **Mem-Absorption spike** expected.                          | Add Cap-Gauge alert at 3.6 bpp; email & Slack webhook.                 |

Deliverable: **v0.3 tag** — Gemma checkpoints 0-4 + dashboard live.

---

### **W-4  —  Replay-Ablation & Minimal Llama-3 Run**

1. **Replay-Ablation task** (async every 50 k steps) coded & queued.
2. **Tiny retrieval DB** (Slim-Pile 10 M docs) + RAG-Trace hooked into Grafana.
3. **Spin up Llama-3-8 B QLoRA run** (flash-attn 2 + 4-bit) straight **with hooks**; save only two checkpoints (start & 40 % tokens).
4. **Ablation grid** (no temporal dropout **vs** dropout) launched on Gemma using stored checkpoints (no extra tokens).
5. **Cap-Gauge reaches ≈3.6 bpp** for Gemma → freeze **checkpoint G-cap**.

---

### **W-5  —  Evaluation, Safety Probe, User Study**

* **Safety probe**: profanity circuit causal-edit test on Gemma (baseline vs prior-alignment).
* **User/auditor study**: 5 colleagues rate dashboard SUS; collect metrics.
* **Aggregate results** (SAEBench, Temporal Purity Gain, Mem-Absorption, Cap-Gauge, edit success).
* **Write Results & Analysis section** while training jobs finish remaining tokens (no new features).

---

### **W-6  —  Wrap-Up & Release**

1. **Final freeze & tags**

   * `v1.0-gemma` → G-cap checkpoint + SAE basis + metrics.
   * `v1.0-llama3` → Llama-3 40 %-tokens checkpoint.
   * `dvc push && git push --tags`.
2. **Paper & artefacts**

   * Write Phase A+B paper (6-page main + appendix).
   * Export Grafana dashboard JSON, WandB CSV, and birth-curve figures to `reports/`.
3. **Repro check**

   ```bash
   git clone ... tmp && cd tmp && dvc pull && pytest -m "smoke"
   ```
4. **Submit** to target venue (or arXiv) + internal hand-off demo.

---

### **Daily Checklist (all 6 weeks)**

* **09:00** — Pull latest `main`, `dvc pull`, scan Grafana alerts.
* **13:00** — Quick WandB glance; Route-SAE refresh if a new checkpoint landed.
* **EOD** — `make freeze CHK=$(last_chk)` if milestone met; `dvc gc --workspace`.

---

### What’s *not* included (parked for after Week 6)

* Full multimodal LLaVA branch.
* Extended ablation grid (hook freq, vision-only, large RAG store).
* ROCm portability tests.

---

Follow this crash-mode schedule and you’ll still deliver **Phase A insights, live Phase B telemetry, and a publishable Gemma case study** inside six weeks. Good luck—time to hit *train()*!
