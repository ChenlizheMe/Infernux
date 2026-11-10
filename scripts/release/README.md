# Hub publication and update discovery

Every public change must have a new version/build identity. Keep the base
version in `pyproject.toml` and `python/infernux/version.py` aligned. A new base
release starts with `ENGINE_BUILD_NUMBER = 1` and Hub `build-number = 1`.
Engine rebuilds increment `ENGINE_BUILD_NUMBER`; Hub-only updates increment
`tool.infernux.hub.build-number` independently. Numeric Hub revisions are
understood by installed Hubs. Published artifact URLs are immutable.

Platform plugin packages have their own versions. Their minimum engine
requirement must match the base engine version used to build their native
payload, and exported Players check the matching runtime contract. Release
packagers derive this version from package metadata; Web CMake derives it
from the engine's version source. Publish a new plugin version when its
payload changes.

After builds and tests, the publisher uses
`build_release_catalog.py --publication-status` to skip publication successfully
when the current Hub version/build is already published. A newer identity proceeds
to publication; a version regression or invalid release metadata still fails.
`--check-version` and catalog generation keep their strict checks against
replacing published identities. Recovery is reserved for a failed gated
publication and bypasses the unchanged-version skip. Public download pages and
catalogs change only after the actual artifacts have been published.

## Release sequence

1. Update the source release identities, both README badges/citations and the
   bilingual changelogs before building; the README is embedded in wheel metadata.
   Build and test the versioned engine using the desktop and Player workflows.
2. Publish each platform plugin's matching precompiled payload from its own
   repository. A workflow-only repair can recover publication of an existing
   tagged payload without changing that tag.
3. Run **Publish Official Plugins**. It reads the exact versions in
   `official-registry.json`, downloads their GitHub release assets, and uploads
   them through the shared R2 publisher. An optional reference publishes one
   package; leaving it empty publishes all official packages. Both modes update
   the complete registry last.
   Every advertised package, including unchanged editor tools, must be available
   before the public registry is updated.
4. Run **Publish Infernux Release** to build, test, and publish. Enable
   `sign_windows_release` only after the release certificate is issued and the
   configuration below is complete.
   Unsigned releases are explicitly disclosed. A prewritten draft requires
   `replace_existing_release=true`.
5. The publisher uploads wheels to PyPI, Hub assets to GitHub/R2, publishes the
   release, and regenerates the website/catalogs from the actual published URLs.
   `build_release_notes.py` reads the English root changelog; `sync_release_site.py`
   updates the website's versioned history after publication.

Official reference ownership is registered in `packaging/official_plugin_repositories.json`.
This includes independent repositories that have not released a package yet;
registration alone does not advertise an installable version.

Independent official releases are recorded in `packaging/official_plugin_releases.json`
using the existing official registry entry format. Catalog generation merges these
releases with the source-backed entries in `external/plugins/plugins.json` and
rejects duplicate references. It does not fetch independent plugin sources or
bundle their archives. Add a release only after its GitHub asset exists; approve
its exact upload destination in the R2 worker before publishing the registry.

### Windows signing setup

SignPath project `Infernux`, organization
`a6c4508c-c43a-4faf-a209-f8198c94a171`, uses two artifact configurations:
`hub` and `installer`. Their exact XML definitions are in `signpath/` beside
this document. The GitHub artifact is a ZIP even when it contains one EXE.
Both configurations restrict the executable name, product name `Infernux`,
and Windows product version. They do not sign third-party DLLs or engine wheels.
`hub` is the default configuration; the obsolete unrestricted `initial`
configuration is inactive. CI always selects its configuration explicitly.

Repository configuration:

- Secret `SIGNPATH_API_TOKEN`: the existing **CI builds** submitter's token.
  Keep it out of source, logs, workflow inputs, and chat transcripts.
- Variable `SIGNPATH_TEST_CERTIFICATE_THUMBPRINT`: the test certificate's SHA-1
  thumbprint, copied from SignPath. This is a public certificate identifier,
  not a signature algorithm selection.
- Variable `SIGNPATH_RELEASE_CERTIFICATE_THUMBPRINT`: the issued release
  certificate's SHA-1 thumbprint. Do not use the test certificate here.
- Link SignPath's `GitHub.com` trusted build system to this project and grant
  `GITHUB_TOKEN` `actions: read` through the workflow permissions. Preserve
  trusted build origin verification and manual release approval. Installing
  the SignPath GitHub App is needed only if audit-log evaluation is required;
  it is not a prerequisite for submitting builds with the GitHub action.

First run **Test Windows Code Signing**. It builds only the Hub/installer,
uses `test-signing`, and retains explicitly labelled test artifacts for seven
days. It has no release, PyPI, R2 or catalog publishing step. On its disposable
Windows runner only, verification temporarily trusts the exact pinned test
certificate and removes that trust afterward. It never changes a user's trust
store. Test signing must pass before enabling a signed public release.

The shared build action uses this ordering:

1. Build the Hub and submit it to SignPath.
2. Verify the pinned certificate, timestamp, product metadata, and preservation
   of the original executable bytes (except Authenticode header fields).
3. Package this signed Hub into both the update ZIP and the installer input;
   assert that both contain the identical executable. Do not rerun the CMake
   installer target afterward: its dependencies rebuild the Hub.
4. Build the installer, then submit and verify the installer signature.
5. Upload the final distribution without further changes.

The current release policy requires the maintainer to approve each of these
two requests. Each request waits for up to one hour. A failure, denial or timeout
stops the build; it does not silently fall back to an unsigned release. No
automated step approves requests on behalf of the maintainer.

Publication independently checks the actual installer and archived Hub on
Windows. It rejects test signatures even if someone selects unsigned publishing.
Only the latest attempt of a successful official desktop build is accepted;
the one-click release may publish its own run after its prerequisite jobs pass.
When recovering, rerun all source build jobs together, not just a failed job,
so Windows and Linux artifacts have the same run-attempt identity.

All channels consume the same final files. GitHub asset SHA-256 digests and
PyPI wheel digests must match the local files. Existing PyPI filenames cannot
silently hide different bytes behind `--skip-existing`. R2 uploads skip identical
existing objects, reject different existing bytes, and verify the actual public
download after uploading. Publication is serialized; do not run independent
upload clients concurrently against the same keys. No existing public release
is re-signed in place: bump the wheel build number for its signed successor.

As of 2026-10-03, the release certificate is **CSR PENDING**. The test certificate
is available; creating the integration does not itself issue the release
certificate. Only SignPath can finish that external step.

### Upload service

`r2-upload-worker.mjs` is the source of `upload.infernux-engine.com`, currently
deployed under the existing Cloudflare Worker name `infernux-r2-upload-040`.
The name is an infrastructure identifier, not the engine release version.
Its `BUCKET` and `UPLOAD_TOKEN` bindings remain server-side. Ordinary plugin
releases use its exact destination/source lists. Engine wheels, versioned Player
plugins and Hub destinations use validated version/build/filename contracts,
so a new engine release or hotfix does not require another wheel allowlist edit. Unauthorized
object paths return HTTP 400 instead of an opaque upload failure.

Deploy source changes through Cloudflare's Worker script-content endpoint,
preserving existing bindings and routing. Never put credentials in this source.
The authenticated `/stat` operation checks storage before an immutable upload;
it must not query the public CDN for a file that does not yet exist. Download
verification waits at most four minutes for an older cached 404 after a confirmed
write. Other HTTP errors and changed contents remain immediate failures.

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

- If all platform acceptance jobs passed but publication stopped with the release
  still a draft, run **Recover Infernux Release Publication** with the original
  run ID and exact source SHA. It verifies every original job and permits only
  publication-file changes in the current checkout. It reuses the original
  distribution attempt, repeats Windows signing verification, and runs the
  publication contract tests. Engine or build changes require a new complete run.
  This explicitly enables draft recovery and preserves immutable PyPI files.
  Before recovery, create the release tag at the exact tested source SHA.
  Recovery checks existing release metadata instead of rewriting it. If a
  maintainer already made that verified release public, recovery verifies its
  immutable assets and completes its catalogs without uploading replacements.
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

For a Hub-only publication prepared locally, `Upload verified Hub release assets`
can transfer the six immutable `hub/<version>/build-<revision>/` objects into the
matching GitHub draft. Supply the draft release ID, frozen source SHA and its
successful Desktop Release And Tests run. Both portable Hub jobs must have passed. The workflow
preserves uploaded files and leaves the release as a draft; publish it and update
the catalogs after upload verification. Engine wheels are unchanged.

The historical audit below predates the 0.4.1 publication work.

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
  --confcutdir=tests/hub tests/hub` — 375 passed, 3 skipped. The
  skips are POSIX permission checks and Linux standalone CPython layout on
  the Windows host. `python -m pytest -q --noconftest
  tests/python/test_release_automation.py` — 3 passed. Workflow YAML parsing and
  `git diff --check` passed. No native engine test fixture was loaded.
- The final suite includes unset/enabled/disabled startup and settings behavior,
  confirmation before any update download, and English/Chinese installer text
  layout. The installer height now follows its content so the disclosure fits.
- A synthetic 0.4.1 catalog retaining the published asset shape was accepted by
  the refreshed 0.4.0 updater on both Windows and Linux, and rejected by the
  initial v0.4.0 updater on both platforms, confirming the compatibility table.
  The new source queried the live catalog with current version 0.3.7 and offered
  the correct full-installer URL for each platform.
