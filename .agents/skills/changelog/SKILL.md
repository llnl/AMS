---
name: changelog
description: >
  Maintain a CHANGELOG.md in Keep a Changelog format: record new features, changes,
  and fixes under an Unreleased section, and cut releases. Use whenever a
  user-facing change ships or the user says "update the changelog", "add a
  changelog entry", "record this feature/fix", "cut a release", "what changed
  since <ref>", or mentions release notes or a CHANGELOG. Works for versioned and
  unversioned projects — fall back to dates plus merge-request or commit IDs when
  there are no version numbers. Trigger on any notable change worth recording, even
  without the word "changelog".
---

# Changelog

Maintain `CHANGELOG.md` at the repo root in **Keep a Changelog** format: a
human-readable, reverse-chronological list of notable changes, grouped by release.
A changelog is for humans — it is not a `git log` dump. The format follows
Keep a Changelog (https://keepachangelog.com).

## Structure

```markdown
# Changelog

## [Unreleased]

### Added
- User-facing description of a new feature (#123).

## [1.2.0] - 2026-07-08

### Added
- ...
### Fixed
- ...
```

- Newest first. Keep an `## [Unreleased]` section at the top as a staging area.
- Dates are ISO 8601 (`YYYY-MM-DD`).
- Group each change under one of six headings; omit headings with no entries:
  - **Added** — new features.
  - **Changed** — changes to existing behavior.
  - **Deprecated** — features slated for removal.
  - **Removed** — features now removed.
  - **Fixed** — bug fixes.
  - **Security** — vulnerability fixes.

Initialize on first use: if `CHANGELOG.md` is missing, create it with the header
above and an empty `## [Unreleased]`. Never overwrite existing history.

## Adding an entry

Add to `## [Unreleased]` when a change merges — not at release time — so nothing
is forgotten.

1. Pick the right group heading (create it under Unreleased if absent).
2. Write one bullet per notable change in **plain, user-facing language**: what
   changed and why it matters, not the commit subject. Rewrite
   "fix: handle keydown in modal (#412)" as "Fixed the dialog not closing on Escape."
3. Reference the source for an audit trail: the PR/merge-request ID (`#123`, `!57`)
   or a short commit SHA when there is no PR. Optionally lead with a bold name:
   `- **CSV export:** feedback entries can now be exported to CSV (#234).`
4. Skip internal churn (refactors, formatting, test-only changes) unless it is
   notable to users or integrators.

Drafting from history is fine — inspect `git log <last-ref>..HEAD` or the merged
PRs — but always curate and rewrite into user-facing wording; never paste raw
commit messages.

## Cutting a release

Move the `## [Unreleased]` entries into a new release section, then leave an empty
`## [Unreleased]` at the top. The release identifier is flexible:

- **Versioned (SemVer):** `## [1.4.0] - YYYY-MM-DD`. Bump MAJOR for breaking
  changes, MINOR for new features, PATCH for fixes.
- **Unversioned:** use a dated header, annotated with the merge request or commit
  that marks the release point:
  ```
  ## [2026-07-08] — mr !57
  ## [2026-07-08] — a1b2c3d
  ```

Optionally, add comparison links at the bottom so headers are clickable. On a git
host these are compare URLs — by tag (`compare/v1.3.0...v1.4.0`) or, for
unversioned projects, by commit/MR range (`compare/<old-sha>...<new-sha>`):

```
[Unreleased]: https://<host>/<repo>/compare/<latest>...HEAD
[1.4.0]: https://<host>/<repo>/compare/v1.3.0...v1.4.0
```

## Conventions

- Write for readers who have never seen the code: no internal ticket shorthand or
  component names without explanation.
- Be brief, we do not want the changelog to be millions of lines long. Not more than
  one sentence for each change.
- Make "update the changelog" part of the definition of done — the person shipping
  the change writes the entry, since they have the context.
- Do not reconstruct a changelog from memory; derive it from git history and curate.
- Do not update past entries of a changelog without very good reason and notify the user.
- If an architecture wiki is maintained (see the `codebase-map` skill), a change
  notable enough for the changelog that also alters structure should update both.