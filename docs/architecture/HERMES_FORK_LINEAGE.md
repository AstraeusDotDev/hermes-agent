# Hermes fork lineage and publishing rule

This repository has two remotes with distinct roles:

| Remote | URL | Role |
|---|---|---|
| `origin` | `https://github.com/NousResearch/hermes-agent.git` | Upstream source for updates and upstream changes. Read-only for Astraeus work; never a publishing target. |
| `fork` | `https://github.com/AstraeusDotDev/hermes-agent.git` | Astraeus integration target. Astraeus work lands on local `main` and is pushed here. |

The remote URLs were read from this checkout's `.git/config`. The measured refs at reconciliation were `origin/main` = `8b66a51036c1e20920a17cdd049fdf55c968d683` and `fork/main` = `708447b9a36595a45821875dd4d79453dc5eb838` (`feat(kanban): retirement fence + worker identity capture`). The local integration base was `main` = `8162d2a0ca` and included the six resource-intent commits plus the receipt commit. The fence commit was not an ancestor of that base; its parent and the merge base were `167448edca5919c4afacb95f10840f8e56f6d647`.

## Why local-only commits move off `main`

`hermes update` first attempts a fast-forward to the selected update remote. If history has diverged, `hermes_cli/update_cmd.py` prints the divergence warning and runs `git reset --hard <merge_ref>` (currently around lines 830–831). Before that reset, the updater attempts to save the pre-update HEAD in an expiring orphan-rescue ref and reports commits leaving the tracked branch. This rescue ref is recovery insurance, not durable integration: the active local branch is reset to the update remote, so unpushed commits disappear from `main` and must be recovered from the rescue ref if needed. This is expected update behavior, not a reason to disable updates. The remedy is to land the commits on the Astraeus fork.

## Reconciled lineage

The fork-only fence could not be applied as a clean patch because upstream had split the old `kanban_db.py` implementation and rewritten the claim paths. The fence was re-applied to the current module layout: schema/migration support in `kanban_db.py` and `kanban_db_connect.py`, worker identity capture in `kanban_db_dispatch.py`, and the original fence tests retained with event-kind assertions. The resource-intent and receipt changes remain in the first-parent line and were not rewritten.

The reconciliation is on `wt/fork-reconcile-001`, based on `8162d2a0ca`:

- `14eb227d0c` — functional retirement-fence reconciliation on the split database modules.
- `bc6c8469f5` — merge commit whose second parent is the original `fork/main` fence commit `708447b9a3`. Its tree keeps the tested refactor reconciliation from the first parent while retaining the fork commit in history. This makes a later fast-forward publication to `fork/main` possible without force-pushing.

The migration was rehearsed on a SQLite backup of the live board, never on the live file. In the isolated copy the two fence columns were removed to simulate the pre-migration schema; `init_db(Path)` was run twice. All 14 table row counts were unchanged, the columns returned as `worker_create_time REAL` and `worker_exe_path TEXT`, and `PRAGMA integrity_check` returned `ok`.

## Standing publish rule

Astraeus changes land on local `main` and publish to `fork`, never `origin`. After Steward review, with the source branch still present and no intervening `main` changes, the expected fast-forward sequence is:

```sh
cd /home/jfensch/.hermes/hermes-agent
git fetch fork
git checkout main
git merge --ff-only wt/fork-reconcile-001
git merge-base --is-ancestor fork/main main || { printf '%s\n' 'fork/main is not an ancestor of main; stop and reconcile'; exit 1; }
git push fork main:main
```

If the ancestry check fails or the fast-forward merge cannot be made, stop and reconcile the refs; do not force-push or redirect the push to `origin`. This task did not push either remote.
