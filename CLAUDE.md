# escape-flu-smh: notes for coding agents

Read [AGENT_HANDOFF.md](AGENT_HANDOFF.md) before changing anything. Its first sections cover where things are, how
the owner works, the history and the pitfalls.

Rules that apply to every session:
- The owner submits all SLURM jobs. Write the sbatch script or command; never run `sbatch`, `scancel` or `scontrol`.
- Commit or push only when asked.
- No data after the round's fitting cutoff (2026-10-03 for Round 1 2026-27) may be used for fitting or selection.
  Newer data is for validation only.
- Do not `git pull` the hub clone in `../flu-round1-2026-27/external/flu-scenario-modeling-hub`. Use `git fetch` and
  `git show origin/main:<path>`.
- Environment: `module load gcc/14.2.0; eval "$($HOME/conda/bin/conda shell.bash hook)"; conda activate escape-sw`.
  Build experiment lists (`$E`) from inside this directory.
- Never edit a model ESL that experiments point at. Add a new version file instead.
- The simulator ignores unknown config names without warning. Check new names against the ESL.
