# Hub publication and update discovery

Every public change to the Hub must increment `project.version`. Changing the
wheel build number or replacing files under the same release version cannot
notify installed Hubs, whose update identity is the application version.
`build_release_catalog.py --check-version` checks this before publication, and
the release workflow allows its recovery switch only for unpublished drafts.
The current repository version remains 0.4.0; the next corrected Hub must use a
new version, such as 0.4.1. No release was published by this audit.

The authoritative update document is
[hub-catalog.json](https://infernux-engine.com/hub-catalog.json). Its asset URLs
prefer Cloudflare R2; the catalog can provide a GitHub asset mirror. If the
catalog's network request fails, Hub tries the same committed document at
[the static source mirror](https://raw.githubusercontent.com/ChenlizheMe/Infernux/master/docs/hub-catalog.json)
once. A valid response, including a current version, remains authoritative.
An invalid document is an error, not an invitation to search unrelated release
APIs. Unpublished stable entries and network failures must not mean "up to date".

The website is a Cloudflare proxy in front of GitHub Pages, configured as
`build_type=legacy`, source `master:/docs`. A `GITHUB_TOKEN` push alone does not
trigger its Pages build, according to
[GitHub's publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
After pushing catalogs, the publisher explicitly requests a build through the
[Pages build API](https://docs.github.com/en/rest/pages/pages#request-a-github-pages-build).
This API accepts GitHub App installation tokens with `pages:write`; the workflow
grants that permission through its calling workflows. It waits for the exact
catalog commit and verifies the public document using the same URL and request
headers as Hub. A bounded wait includes propagation of the site's ten-minute
cache. No cache-busting URL or mirror response substitutes for that check.

## Recovery after partial publication

Do not rerun the complete publisher against an already public tag: that would
replace Hub files while leaving installed clients on the same version.

- If the release is still a draft, use the explicit recovery switch.
- If release assets are public but the catalog commit failed, retain or retrieve
  the exact published assets and regenerate only the catalogs using
  `build_release_catalog.py --release-dir <original-assets> --published-at
  <actual-publication-time> --resolve-pypi`. Use the original release's project
  version and build number. The version check permits this when the committed
  catalog still advertises the previous release. Commit/push only the catalog
  documents, then deploy them. Do not rebuild, sign, or upload replacement Hub
  assets during this recovery.
- If the catalog is committed but Pages deployment failed, rerun only
  `deploy_hub_catalog.py --repository ChenlizheMe/Infernux --commit <catalog-commit>
  --catalog docs/hub-catalog.json` from that catalog checkout. This does not
  publish or overwrite release artifacts and does not require another version.

These are release-operator procedures; the audit did not execute their writes.
The new workflow's real publication/deployment path still needs acceptance on
the next release.

## Existing client compatibility

| Installed updater | Behavior and migration boundary |
| --- | --- |
| v0.3.7 | Requires GitHub latest-release assets `SHA256SUMS.txt` and `InfernuxHub-manifest.json`. Their absence returns no update. Current 0.4.0 assets use platform manifests, so use the full installer. |
| Initial v0.4.0 tag | Requires exactly `name`/`url` in installer and manifest assets. Current catalogs also include `fallback_url`, so a higher target version fails its strict schema. Use the full installer. |
| Refreshed v0.4.0 / build-2 updater | Reads the current catalog contract, but cannot distinguish another Hub published under the same 0.4.0 version. A higher application version using that contract is discoverable. |
| Updated source | Preserves that catalog shape, accepts optional asset mirror metadata, orders release candidates correctly, and joins a manual check to any running startup check with a visible result. |

Source edits cannot modify already shipped binaries. The user's exact old Hub
version was not supplied, so this table records reproduced mechanisms rather
than claiming which binary they had. Automatic startup checks and release notices
are enabled when no preference has been saved. A user's explicitly disabled
setting is preserved. Checks only offer an update; downloading and installing
still require confirmation. The manual settings action checks explicitly.

## Read-only audit evidence, 2026-09-13

- The website and the static source catalog both returned HTTP 200 and advertised
  stable 0.4.0, published at `2026-09-06T17:56:54Z`, with Cloudflare `build-2`
  URLs. The website returned `Cache-Control: max-age=600` and
  `Last-Modified: Tue, 08 Sep 2026 03:37:45 GMT`.
- HEAD requests to Cloudflare Windows/Linux update archives returned HTTP 200
  with lengths 32,778,654 / 59,825,838 bytes, exactly matching the catalog. The
  archives were not downloaded.
- The public [0.4.0 asset listing](https://github.com/ChenlizheMe/Infernux/releases/expanded_assets/v0.4.0)
  returned HTTP 200 and contained platform manifests, with no SHA256SUMS or old
  unqualified manifest. The unauthenticated GitHub latest-release API returned
  HTTP 403 rate limit exceeded during the same audit; that is an additional
  network problem, not the explanation for the reproduced schema failures.
- Read-only authenticated Pages queries returned `build_type=legacy`,
  `master:/docs`, and the latest successful build at `2026-09-08T03:36:57Z`,
  commit `d0eac0f2d29c0e606288e5f2d64b5b2fd30ae203`.
- Executing the historical `git show v0.3.7:packaging/hub_updater.py` against the
  current public asset names returned `None`. Executing the initial v0.4.0 and
  pre-fix HEAD updaters against current build-2 metadata with installed version
  0.4.0 returned `UP_TO_DATE`. The initial v0.4.0 code also rejected the current
  catalog's installer metadata when exercising its newer-version branch.
- The first regression run, before implementation, produced 12 failures and
  24 passes, covering the unavailable-source error loss, release-candidate
  comparison, manual/startup race, unpublished catalog, mirror policy, and new
  publication restrictions. Subsequent tests exercise both desktop platforms
  and mock all publication/deployment network writes.
- Final full Hub suite: `python -m pytest -q -rs -o pythonpath=packaging
  --confcutdir=packaging/tests packaging/tests` — 375 passed, 3 skipped. The
  skips are POSIX permission checks and Linux standalone CPython layout on
  the Windows host. `python -m pytest -q --noconftest
  python/test/test_release_automation.py` — 3 passed. Workflow YAML parsing and
  `git diff --check` passed. No native engine test fixture was loaded.
- The final suite includes unset/enabled/disabled startup and settings behavior,
  confirmation before any update download, and English/Chinese installer text
  layout. The installer height now follows its content so the disclosure fits.
- A synthetic 0.4.1 catalog retaining the published asset shape was accepted by
  the refreshed 0.4.0 updater on both Windows and Linux, and rejected by the
  initial v0.4.0 updater on both platforms, confirming the compatibility table.
  The new source queried the live catalog with current version 0.3.7 and offered
  the correct full-installer URL for each platform.
