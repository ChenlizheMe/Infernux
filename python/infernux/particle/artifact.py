"""Save-time Particle AOT artifact registry with last-known-good publication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import Future as _Future
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping

from infernux.engine.path_utils import (
    path_key,
    portable_path,
    relative_path,
    resolved_path,
)
from .asset import ParticleGraphAsset
from .hir import ParticleGraphCompiler, ParticleProgramHIR
from .kernel_ir import ParticleKernelLowerer, ParticleKernelProgram
from .runtime_metadata import decode_particle_runtime_metadata
from .gpu_glsl_backend import (
    GpuParticleGlslLowerer,
    compile_gpu_particle_spirv,
    decode_gpu_particle_spirv,
    validate_gpu_particle_spirv,
)
from .script import ParticleScriptCompiler


PARTICLE_ARTIFACT_SCHEMA = "infernux.particle_artifact"
PARTICLE_RUNTIME_INDEX_SCHEMA = "infernux.particle_runtime_index"
PARTICLE_RUNTIME_INDEX_FILENAME = "RuntimeIndex.json"


def particle_artifact_filename(identity: str) -> str:
    """Return ``{guid}.inxparticle``, matching AssetDatabase runtime paths.

    The identity is the AssetDatabase / ``.meta`` GUID. Graph ``stable_id`` is
    only used when a compile has no asset identity yet (tests or first save).
    """

    token = str(identity or "").strip()
    if not token or not all(character.isalnum() or character in "-_" for character in token):
        token = hashlib.sha256(token.encode("utf-8")).hexdigest()[:32]
    return f"{token}.inxparticle"


class ParticleArtifactError(ValueError):
    pass


class ParticleArtifactSuperseded(ParticleArtifactError):
    """The prepared save no longer owns its source publication."""


@dataclass(frozen=True)
class ParticleArtifact:
    source_key: str
    source_hash: str
    source_kind: str
    revision: int
    semantic_hash: str
    behavior_hash: str
    artifact_path: str
    hir: Mapping[str, Any]
    kernel_ir: Mapping[str, Any]
    gpu_glsl: Mapping[str, Any]
    gpu_spirv: Mapping[str, Any]


@dataclass(frozen=True)
class _CompiledParticleArtifact:
    source_hash: str
    source_kind: str
    stable_id: str
    semantic_hash: str
    behavior_hash: str
    hir: Mapping[str, Any]
    kernel_ir: Mapping[str, Any]
    gpu_glsl: Mapping[str, Any]
    gpu_spirv: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedParticleGraphArtifact:
    """Side-effect-free graph compilation ready for durable publication."""

    compiled: _CompiledParticleArtifact
    source_path: str
    source_key: str
    path_identity: str
    request_generation: int
    artifact_path: str
    source_text: str
    guid: str = ""


@dataclass(frozen=True)
class _ParticlePublication:
    artifact: ParticleArtifact
    source_write: Any = None


class ParticleGraphWriteTicket:
    """Completion of the entire source/artifact/index commit, not just source IO."""

    def __init__(self, future: _Future, prepared: PreparedParticleGraphArtifact):
        self._future = future
        self.path = prepared.source_path
        self.generation = prepared.request_generation

    @property
    def is_complete(self) -> bool:
        return self._future.done()

    @property
    def status(self) -> str:
        if self._future.cancelled():
            return "cancelled"
        if not self.is_complete:
            return "pending"
        error = self._future.exception()
        if isinstance(error, ParticleArtifactSuperseded):
            return "superseded"
        return "failed" if error is not None else "succeeded"

    @property
    def error(self) -> str:
        if self._future.cancelled():
            return "particle graph save was cancelled"
        return str(self._future.exception() or "") if self.is_complete else ""

    @property
    def committed_file_state(self):
        if self.status != "succeeded":
            return None
        return self._future.result().source_write.committed_file_state

    def wait(self) -> None:
        self._future.result()


class ParticleArtifactRegistry:
    _artifacts: dict[str, ParticleArtifact] = {}
    # Runtime decoding is immutable and shared by every component referencing
    # the same AOT revision.  Instance parameters, controllers and playback
    # state deliberately remain on ParticleSystem.
    _runtime_decode_cache: dict[
        tuple[str, str, int],
        tuple[Any, ParticleKernelProgram, tuple[Mapping[str, Any], ...]],
    ] = {}
    _revision = 0
    _request_generation: dict[str, int] = {}
    _next_request_generation = 0
    _lock = threading.RLock()
    # Serialize durable publications without holding the runtime-reader lock
    # while waiting for disk. Request acceptance uses this same commit gate.
    _publication_lock = threading.RLock()
    # Cooked runtime readers never import the editor's thread-pool backend.
    _save_executor = None

    @classmethod
    def clear(cls) -> None:
        with cls._publication_lock, cls._lock:
            cls._artifacts.clear()
            cls._runtime_decode_cache.clear()
            cls._request_generation.clear()
            cls._revision = 0

    @classmethod
    def decode_runtime_artifact(
        cls, artifact: ParticleArtifact
    ) -> tuple[Any, ParticleKernelProgram, tuple[Mapping[str, Any], ...]]:
        """Decode one immutable AOT revision once for all scene instances."""
        if not isinstance(artifact, ParticleArtifact):
            raise TypeError("particle runtime artifact has an invalid type")
        key = (artifact.source_hash, artifact.behavior_hash, artifact.revision)
        with cls._lock:
            cached = cls._runtime_decode_cache.get(key)
        if cached is not None:
            return cached

        metadata = decode_particle_runtime_metadata(artifact.hir)
        kernel = ParticleKernelProgram.from_dict(artifact.kernel_ir)
        decoded_emitters = tuple(
            decode_gpu_particle_spirv(artifact.gpu_spirv, index)
            for index in range(len(metadata.emitters))
        )
        decoded = (metadata, kernel, decoded_emitters)
        with cls._lock:
            published = cls._runtime_decode_cache.setdefault(key, decoded)
            # Hot reload can create many revisions during a long editor run.
            # Retain a bounded working set without adding another lifecycle.
            while len(cls._runtime_decode_cache) > 64:
                oldest = next(iter(cls._runtime_decode_cache))
                if oldest == key and len(cls._runtime_decode_cache) == 1:
                    break
                cls._runtime_decode_cache.pop(oldest, None)
            return published

    @classmethod
    def on_asset_mutation(cls, change) -> None:
        from infernux.engine.interaction import AssetMutationKind, iter_asset_mutations

        for mutation in iter_asset_mutations(change):
            if mutation.kind is not AssetMutationKind.MOVED:
                continue
            destination = mutation.destination_path
            if not destination.lower().endswith((".particlegraph", ".particle.py")):
                continue
            cls.remap_source(
                mutation.source_path,
                destination,
                guid=mutation.guid,
            )

    @classmethod
    def remap_source(cls, source_path: str, destination_path: str, *, guid: str = "") -> None:
        """Rekey live and shipped AOT lookup state after a GUID-stable move."""
        with cls._publication_lock:
            cls._remap_source_under_publication_lock(source_path, destination_path, guid=guid)

    @classmethod
    def _remap_source_under_publication_lock(cls, source_path, destination_path, *, guid):
        source = resolved_path(source_path)
        destination = resolved_path(destination_path)
        source_key = cls._source_key(source)
        destination_key = cls._source_key(destination)
        guid_key = cls._source_key("", guid) if guid else ""
        with cls._lock:
            artifact = (
                cls._artifacts.get(guid_key)
                if guid_key
                else None
            ) or cls._artifacts.get(source_key)
            cls._artifacts.pop(source_key, None)
            # Prepared writes still contain their original target path. A move
            # retires pending writes at both names instead of lending the old
            # path's authority to a different source or destination occupant.
            cls._request_generation.pop(source_key, None)
            cls._request_generation.pop(destination_key, None)
            if artifact is not None:
                next_key = guid_key or destination_key
                moved = replace(artifact, source_key=next_key)
                cls._register_unlocked(moved, next_key, destination_key)

        from infernux.core.document_store import write_document_text
        from infernux.engine.project_context import get_project_root

        project_root = get_project_root()
        if not project_root:
            return
        try:
            old_hint = portable_path(relative_path(source, project_root, resolve=False))
            new_hint = portable_path(
                relative_path(destination, project_root, resolve=False)
            )
        except ValueError:
            return
        index_path = os.path.join(
            project_root,
            "Library",
            "Artifacts",
            "Particle",
            PARTICLE_RUNTIME_INDEX_FILENAME,
        )
        try:
            index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if (
            type(index) is not dict
            or index.get("$schema") != PARTICLE_RUNTIME_INDEX_SCHEMA
            or type(index.get("entries")) is not list
        ):
            return

        old_identity = portable_path(old_hint).casefold()
        changed = False
        entries: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for value in index["entries"]:
            if (
                type(value) is not dict
                or set(value) != {"guid", "path_hint", "stable_id"}
                or any(type(value.get(key)) is not str for key in value)
            ):
                continue
            entry = dict(value)
            if (
                (guid and entry["guid"] == guid)
                or portable_path(entry["path_hint"]).casefold() == old_identity
            ):
                entry["path_hint"] = new_hint
                if guid:
                    entry["guid"] = guid
                changed = True
            identity = (entry["guid"], entry["stable_id"])
            if identity in seen:
                changed = True
                continue
            seen.add(identity)
            entries.append(entry)
        if not changed:
            return
        entries.sort(key=lambda entry: portable_path(entry["path_hint"]).casefold())
        write_document_text(
            index_path,
            json.dumps(
                {"$schema": PARTICLE_RUNTIME_INDEX_SCHEMA, "entries": entries},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    @classmethod
    def get(cls, path: str = "", *, guid: str = "") -> ParticleArtifact | None:
        with cls._lock:
            return cls._current_unlocked(path, guid)

    @classmethod
    def iter_project_sources(cls, project_root: str) -> list[tuple[str, str]]:
        """Return ``(source_path, guid)`` for every particle authoring source."""
        root = resolved_path(project_root)
        if not root:
            return []
        assets_root = os.path.join(root, "Assets")
        if not os.path.isdir(assets_root):
            return []
        sources: list[tuple[str, str]] = []
        for root, directories, filenames in os.walk(assets_root):
            directories[:] = [
                name for name in directories if not name.startswith(".")
            ]
            for filename in filenames:
                lower = filename.casefold()
                if not (
                    lower.endswith(".particlegraph") or lower.endswith(".particle.py")
                ):
                    continue
                source_path = resolved_path(os.path.join(root, filename))
                sources.append((source_path, cls._resolve_source_guid(source_path)))
        return sources

    @classmethod
    def source_needs_compile(cls, source_path: str, *, guid: str = "") -> bool:
        """Return True when the Library artifact is missing or stale.

        Source identity alone is not enough: generated GPU source and binary
        contracts evolve with the particle compiler.  Validate the complete
        persisted product so a build cannot ship an artifact produced by an
        older lowering implementation after an engine upgrade.
        """
        owner = cls._resolve_source_guid(source_path, guid)
        try:
            source = Path(source_path).read_text(encoding="utf-8")
            if source_path.casefold().endswith(".particle.py"):
                current = cls._text_source_hash(source)
                stable_id = (
                    ParticleScriptCompiler()
                    .parse(source, source_name=source_path)
                    .stable_id
                )
            else:
                graph = ParticleGraphAsset.from_json(source)
                current = cls._graph_source_hash(graph)
                stable_id = graph.stable_id
        except (
            OSError,
            UnicodeDecodeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return True
        artifact_path = cls._existing_artifact_path(
            guid=owner, source_path=source_path, stable_id=stable_id
        ) or cls._artifact_path(owner or stable_id)
        if not artifact_path or not os.path.isfile(artifact_path):
            return True
        return (
            cls._load_persisted(
                artifact_path,
                key=cls._source_key(source_path, owner),
                source_hash=current,
                source_kind=(
                    "script"
                    if source_path.casefold().endswith(".particle.py")
                    else "graph"
                ),
            )
            is None
        )

    @classmethod
    def ensure_source_compiled(
        cls, source_path: str, *, guid: str = ""
    ) -> ParticleArtifact | None:
        """Compile one source when its Library artifact is missing or stale."""
        if not cls.source_needs_compile(source_path, guid=guid):
            return cls.get(source_path, guid=guid)
        return cls.compile_path(source_path, guid=guid, force_recompile=True)

    @classmethod
    def ensure_project_compiled(
        cls,
        project_root: str,
        *,
        raise_on_error: bool = False,
    ) -> dict[str, list[str]]:
        """Fill missing or stale particle artifacts for every project source.

        Editor startup logs compile failures so the session can continue.
        Player cook raises so a missing product cannot ship.
        """
        from infernux.engine.project_context import using_project_root
        from infernux.debug import Debug

        compiled: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        root = resolved_path(project_root)
        if not root:
            return {"compiled": compiled, "skipped": skipped, "failed": failed}
        with using_project_root(root):
            for source_path, guid in cls.iter_project_sources(root):
                try:
                    if not cls.source_needs_compile(source_path, guid=guid):
                        skipped.append(source_path)
                        continue
                    cls.compile_path(source_path, guid=guid, force_recompile=True)
                    compiled.append(source_path)
                except Exception as exc:
                    failed.append(source_path)
                    message = (
                        f"Particle artifact compile failed: {source_path}: {exc}"
                    )
                    if raise_on_error:
                        raise ParticleArtifactError(message) from exc
                    Debug.log_error(message)
        return {"compiled": compiled, "skipped": skipped, "failed": failed}

    @classmethod
    def load_runtime_reference(
        cls, *, guid: str
    ) -> ParticleArtifact | None:
        """Load a shipped AOT artifact by its imported asset GUID."""
        from infernux.engine.project_context import get_project_root

        project_root = get_project_root()
        if not project_root:
            return None
        artifact_root = os.path.join(
            project_root, "Library", "Artifacts", "Particle"
        )
        index_path = os.path.join(artifact_root, PARTICLE_RUNTIME_INDEX_FILENAME)
        if not os.path.isfile(index_path):
            return None
        try:
            index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParticleArtifactError(
                f"particle runtime index cannot be read: {index_path}"
            ) from exc
        if (
            type(index) is not dict
            or index.get("$schema") != PARTICLE_RUNTIME_INDEX_SCHEMA
            or type(index.get("entries")) is not list
        ):
            raise ParticleArtifactError("particle runtime index is not current")

        wanted_guid = str(guid or "").strip()
        if not wanted_guid:
            raise ParticleArtifactError(
                "shipped particle artifact lookup requires a non-empty GUID"
            )
        selected = None
        for entry in index["entries"]:
            if type(entry) is not dict or type(entry.get("guid")) is not str:
                continue
            entry_guid = entry["guid"].strip()
            if entry_guid == wanted_guid:
                selected = entry
                break
        if selected is None:
            return None

        stable_id = selected.get("stable_id", "")
        if not stable_id or not all(
            character.isalnum() or character in "-_" for character in stable_id
        ):
            raise ParticleArtifactError(
                f"particle runtime index has an invalid stable_id: {stable_id!r}"
            )
        artifact_path = os.path.join(
            artifact_root,
            particle_artifact_filename(wanted_guid),
        )
        try:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            source_hash = payload["source_hash"]
            source_kind = payload["source_kind"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ParticleArtifactError(
                f"shipped particle artifact cannot be read: {artifact_path}"
            ) from exc
        key = cls._source_key("", guid)
        artifact = cls._load_persisted(
            artifact_path,
            key=key,
            source_hash=source_hash,
            source_kind=source_kind,
        )
        if artifact is None:
            raise ParticleArtifactError(
                f"shipped particle artifact is invalid: {artifact_path}"
            )
        with cls._lock:
            cls._register_unlocked(artifact, key, key)
            cls._revision = max(cls._revision, artifact.revision)
        return artifact

    @classmethod
    def save_graph_asset(
        cls,
        asset: ParticleGraphAsset,
        path: str,
        *,
        guid: str = "",
        expected_file_state=None,
    ) -> ParticleArtifact:
        """Compile and commit one graph snapshot under its request ownership."""
        prepared = cls.prepare_graph_asset(asset, path, guid=guid)
        return cls._commit_prepared_graph(prepared, expected_file_state=expected_file_state).artifact

    @classmethod
    def prepare_graph_asset(
        cls,
        asset: ParticleGraphAsset,
        path: str,
        *,
        guid: str = "",
    ) -> PreparedParticleGraphArtifact:
        """Compile one immutable graph snapshot without touching disk or runtime."""
        if not isinstance(asset, ParticleGraphAsset):
            raise ParticleArtifactError("particle graph must be a ParticleGraphAsset")
        source_path = resolved_path(path)
        if not source_path:
            raise ParticleArtifactError("particle graph save path cannot be empty")
        source_hash = cls._graph_source_hash(asset)
        path_identity = cls._source_key(source_path)
        key = cls._source_key(source_path, guid)
        ticket = cls._begin_request(path_identity)

        with cls._lock:
            existing = cls._current_unlocked(source_path, guid)
        if existing is not None and existing.source_hash == source_hash:
            compiled = cls._compiled_from_artifact(existing, stable_id=asset.stable_id)
        else:
            try:
                compiled = cls._compile_graph_asset(asset, source_path)
            except Exception as exc:
                raise ParticleArtifactError(
                    f"particle graph AOT compile failed: {type(exc).__name__}: {exc}"
                ) from exc

        owner = cls._resolve_source_guid(source_path, guid)
        artifact_path = cls._artifact_path(owner or asset.stable_id)
        source_text = (
            json.dumps(asset.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        return PreparedParticleGraphArtifact(
            compiled,
            source_path,
            key,
            path_identity,
            ticket,
            artifact_path,
            source_text,
            owner,
        )

    @classmethod
    def submit_prepared_graph(
        cls,
        prepared: PreparedParticleGraphArtifact,
        *,
        expected_file_state=None,
    ) -> ParticleGraphWriteTicket:
        """Commit a compiled editor snapshot in the background with the same save gate."""
        if not isinstance(prepared, PreparedParticleGraphArtifact):
            raise TypeError("prepared particle graph artifact has an invalid type")
        with cls._lock:
            if cls._save_executor is None:
                from concurrent.futures import ThreadPoolExecutor

                cls._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="particle-save")
            future = cls._save_executor.submit(
                cls._commit_prepared_graph, prepared, expected_file_state=expected_file_state,
            )
        return ParticleGraphWriteTicket(future, prepared)

    @classmethod
    def require_current_graph_publication(cls, prepared: PreparedParticleGraphArtifact) -> None:
        """Reject stale editor completion before updating its view or save point."""
        with cls._lock:
            if cls._request_generation.get(prepared.path_identity) != prepared.request_generation:
                raise ParticleArtifactSuperseded("particle graph save was superseded before editor publication")

    @classmethod
    def _commit_prepared_graph(cls, prepared, *, expected_file_state=None) -> _ParticlePublication:
        return cls._publish_compiled(
            prepared.compiled,
            source_path=prepared.source_path,
            key=prepared.source_key,
            path_identity=prepared.path_identity,
            ticket=prepared.request_generation,
            artifact_path=prepared.artifact_path,
            source_text=prepared.source_text,
            expected_file_state=expected_file_state,
            guid=prepared.guid,
        )

    @classmethod
    def compile_path(
        cls,
        path: str,
        *,
        guid: str = "",
        force_recompile: bool = False,
        runtime_artifact_path: str = "",
    ) -> ParticleArtifact:
        source_path = resolved_path(path)
        try:
            source = Path(source_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ParticleArtifactError(f"failed to read particle source: {exc}") from exc
        source_kind = "script" if source_path.lower().endswith(".particle.py") else "graph"
        path_identity = cls._source_key(source_path)
        key = cls._source_key(source_path, guid)
        ticket = cls._begin_request(path_identity)

        try:
            graph_asset = (
                ParticleGraphAsset.from_json(source) if source_kind == "graph" else None
            )
            source_hash = (
                cls._graph_source_hash(graph_asset)
                if graph_asset is not None
                else cls._text_source_hash(source)
            )
            stable_id = (
                graph_asset.stable_id
                if graph_asset is not None
                else ParticleScriptCompiler()
                .parse(source, source_name=source_path)
                .stable_id
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ParticleArtifactError(f"particle AOT compile failed: {exc}") from exc

        with cls._lock:
            existing = cls._current_unlocked(source_path, guid)
            if (
                not force_recompile
                and cls._request_generation.get(path_identity) == ticket
                and existing is not None
                and existing.source_hash == source_hash
                and bool(existing.artifact_path)
            ):
                alias = replace(existing, source_key=key)
                cls._register_unlocked(alias, key, path_identity)
                return alias

        owner = cls._resolve_source_guid(source_path, guid)
        publish_path = (
            resolved_path(runtime_artifact_path)
            if str(runtime_artifact_path or "").strip()
            else cls._artifact_path(owner or stable_id)
        )
        artifact_path = (
            cls._existing_artifact_path(
                guid=owner, source_path=source_path, stable_id=stable_id
            )
            or publish_path
        )
        if not force_recompile and existing is None and artifact_path:
            persisted = cls._load_persisted(
                artifact_path,
                key=key,
                source_hash=source_hash,
                source_kind=source_kind,
            )
            if persisted is not None:
                with cls._lock:
                    if cls._request_generation.get(path_identity) != ticket:
                        return cls._superseded_unlocked(source_path, guid)
                    cls._register_unlocked(persisted, key, path_identity)
                    cls._revision = max(cls._revision, persisted.revision)
                    return persisted

        if not force_recompile and existing is not None and existing.source_hash == source_hash:
            compiled = cls._compiled_from_artifact(existing, stable_id=stable_id)
        else:
            try:
                compiled = (
                    cls._compile_script_source(source, source_path)
                    if graph_asset is None
                    else cls._compile_graph_asset(graph_asset, source_path)
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ParticleArtifactError(f"particle AOT compile failed: {exc}") from exc
        return cls._publish_compiled(
            compiled,
            source_path=source_path,
            key=key,
            path_identity=path_identity,
            ticket=ticket,
            artifact_path=publish_path,
            guid=owner,
        ).artifact

    @classmethod
    def _compile_graph_asset(
        cls, asset: ParticleGraphAsset, source_path: str
    ) -> _CompiledParticleArtifact:
        program = ParticleGraphCompiler().compile(asset, source_name=source_path)
        return cls._lower_program(
            program,
            source_hash=cls._graph_source_hash(asset),
            source_kind="graph",
            stable_id=asset.stable_id,
        )

    @classmethod
    def _compile_script_source(
        cls, source: str, source_path: str
    ) -> _CompiledParticleArtifact:
        program = ParticleScriptCompiler().compile(source, source_name=source_path)
        return cls._lower_program(
            program,
            source_hash=cls._text_source_hash(source),
            source_kind="script",
            stable_id=program.stable_id,
        )

    @staticmethod
    def _lower_program(
        program: ParticleProgramHIR,
        *,
        source_hash: str,
        source_kind: str,
        stable_id: str,
    ) -> _CompiledParticleArtifact:
        hir = _program_to_dict(program)
        kernel_program = ParticleKernelLowerer().lower(program)
        kernel_ir = kernel_program.to_dict()
        gpu_program = GpuParticleGlslLowerer().lower(kernel_program)
        gpu_glsl = gpu_program.to_dict()
        gpu_spirv = compile_gpu_particle_spirv(gpu_program)
        return _CompiledParticleArtifact(
            source_hash,
            source_kind,
            stable_id,
            program.semantic_hash,
            program.behavior_hash,
            hir,
            kernel_ir,
            gpu_glsl,
            gpu_spirv,
        )

    @staticmethod
    def _compiled_from_artifact(
        artifact: ParticleArtifact, *, stable_id: str
    ) -> _CompiledParticleArtifact:
        return _CompiledParticleArtifact(
            artifact.source_hash,
            artifact.source_kind,
            stable_id,
            artifact.semantic_hash,
            artifact.behavior_hash,
            artifact.hir,
            artifact.kernel_ir,
            artifact.gpu_glsl,
            artifact.gpu_spirv,
        )

    @classmethod
    def _publish_compiled(
        cls,
        compiled: _CompiledParticleArtifact,
        *,
        source_path: str,
        key: str,
        path_identity: str,
        ticket: int,
        artifact_path: str,
        source_text: str | None = None,
        expected_file_state=None,
        guid: str = "",
    ) -> _ParticlePublication:
        with cls._publication_lock:
            with cls._lock:
                if cls._request_generation.get(path_identity) != ticket:
                    if source_text is not None:
                        raise ParticleArtifactSuperseded(
                            f"particle graph save for {source_path!r} was superseded by a newer request"
                        )
                    return _ParticlePublication(cls._superseded_unlocked(source_path, key))

                current = cls._artifacts.get(key) or cls._artifacts.get(path_identity)
                if current is not None and current.source_hash == compiled.source_hash:
                    revision = current.revision
                else:
                    revision = cls._revision + 1

            artifact = ParticleArtifact(
                key,
                compiled.source_hash,
                compiled.source_kind,
                revision,
                compiled.semantic_hash,
                compiled.behavior_hash,
                artifact_path,
                compiled.hir,
                compiled.kernel_ir,
                compiled.gpu_glsl,
                compiled.gpu_spirv,
            )

            from infernux.core.document_store import submit_document_text, write_document_text

            source_write = None
            if source_text is not None:
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                source_write = submit_document_text(
                    source_path, source_text, expected_file_state=expected_file_state,
                )
                source_write.wait()
            if artifact_path:
                os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
                write_document_text(
                    artifact_path,
                    json.dumps(
                        cls._artifact_payload(artifact),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            if compiled.source_kind == "graph":
                cls._publish_runtime_index_entry(
                    source_path,
                    stable_id=compiled.stable_id,
                    guid=guid,
                )

            with cls._lock:
                cls._revision = max(cls._revision, revision)
                cls._register_unlocked(artifact, key, path_identity)
            return _ParticlePublication(artifact, source_write)

    @staticmethod
    def _publish_runtime_index_entry(
        source_path: str, *, stable_id: str, guid: str = ""
    ) -> None:
        from infernux.core.document_store import write_document_text
        from infernux.engine.project_context import get_project_root

        project_root = get_project_root()
        if not project_root:
            return
        try:
            path_hint = portable_path(
                relative_path(source_path, project_root, resolve=False)
            )
        except ValueError:
            return

        guid = ParticleArtifactRegistry._resolve_source_guid(source_path, guid)

        artifact_root = os.path.join(
            project_root, "Library", "Artifacts", "Particle"
        )
        index_path = os.path.join(artifact_root, PARTICLE_RUNTIME_INDEX_FILENAME)
        entries: list[dict[str, str]] = []
        try:
            current = json.loads(Path(index_path).read_text(encoding="utf-8"))
            if (
                type(current) is dict
                and current.get("$schema") == PARTICLE_RUNTIME_INDEX_SCHEMA
                and type(current.get("entries")) is list
            ):
                entries = [
                    entry
                    for entry in current["entries"]
                    if type(entry) is dict
                    and set(entry) == {"guid", "path_hint", "stable_id"}
                    and all(type(entry.get(key)) is str for key in entry)
                ]
        except (OSError, json.JSONDecodeError):
            pass

        path_identity = portable_path(path_hint).casefold()
        entries = [
            entry
            for entry in entries
            if portable_path(entry["path_hint"]).casefold() != path_identity
            and (not guid or entry["guid"] != guid)
        ]
        entries.append(
            {"guid": guid, "path_hint": path_hint, "stable_id": stable_id}
        )
        entries.sort(key=lambda entry: portable_path(entry["path_hint"]).casefold())
        os.makedirs(artifact_root, exist_ok=True)
        write_document_text(
            index_path,
            json.dumps(
                {"$schema": PARTICLE_RUNTIME_INDEX_SCHEMA, "entries": entries},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    @classmethod
    def _begin_request(cls, path_identity: str) -> int:
        with cls._publication_lock, cls._lock:
            # clear/remap must never recycle a ticket held by an in-flight compile.
            cls._next_request_generation += 1
            ticket = cls._next_request_generation
            cls._request_generation[path_identity] = ticket
            return ticket

    @classmethod
    def _current_unlocked(
        cls, path: str, guid: str = ""
    ) -> ParticleArtifact | None:
        return cls._artifacts.get(cls._source_key(path, guid)) or cls._artifacts.get(
            cls._source_key(path)
        )

    @classmethod
    def _register_unlocked(
        cls, artifact: ParticleArtifact, key: str, path_identity: str
    ) -> None:
        cls._artifacts[key] = artifact
        cls._artifacts[path_identity] = artifact

    @classmethod
    def _superseded_unlocked(
        cls, source_path: str, guid_or_key: str = ""
    ) -> ParticleArtifact:
        current = cls._artifacts.get(guid_or_key) or cls._current_unlocked(source_path)
        if current is not None:
            return current
        raise ParticleArtifactError(
            f"particle compile for {source_path!r} was superseded by a newer request"
        )

    @staticmethod
    def _graph_source_hash(asset: ParticleGraphAsset) -> str:
        return hashlib.sha256(asset.canonical_json().encode("utf-8")).hexdigest()

    @staticmethod
    def _text_source_hash(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @staticmethod
    def _artifact_payload(artifact: ParticleArtifact) -> dict[str, Any]:
        return {
            "$schema": PARTICLE_ARTIFACT_SCHEMA,
            "source_key": artifact.source_key,
            "source_hash": artifact.source_hash,
            "source_kind": artifact.source_kind,
            "revision": artifact.revision,
            "semantic_hash": artifact.semantic_hash,
            "behavior_hash": artifact.behavior_hash,
            "hir": artifact.hir,
            "kernel_ir": artifact.kernel_ir,
            "gpu_glsl": artifact.gpu_glsl,
            "gpu_spirv": artifact.gpu_spirv,
        }

    @staticmethod
    def _source_key(path: str, guid: str = "") -> str:
        identity = str(guid or "").strip()
        return identity or path_key(path)

    @classmethod
    def _resolve_source_guid(cls, source_path: str, guid: str = "") -> str:
        """Resolve the stable asset GUID; never invent a new one.

        Texture/Mesh identity comes from ``InxResourceMeta::GenerateGuid()``
        and is preserved in the ``.meta`` sidecar. Particle artifacts use
        that same GUID so copies of a graph cannot share one Library file.
        """
        owner = str(guid or "").strip()
        if owner:
            return owner
        from infernux.core.asset_types import read_meta_guid

        owner = read_meta_guid(source_path)
        if owner:
            return owner
        try:
            from infernux.core.assets import AssetManager

            owner = str(AssetManager._get_guid_from_path(source_path) or "").strip()
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
            return ""
        return owner

    @classmethod
    def _existing_artifact_path(
        cls,
        *,
        guid: str = "",
        source_path: str = "",
        stable_id: str = "",
    ) -> str:
        owner = cls._resolve_source_guid(source_path, guid)
        if owner:
            candidate = cls._artifact_path(owner)
            return candidate if candidate and os.path.isfile(candidate) else ""
        if stable_id:
            candidate = cls._artifact_path(stable_id)
            return candidate if candidate and os.path.isfile(candidate) else ""
        return ""

    @staticmethod
    def _artifact_path(identity: str) -> str:
        from infernux.engine.project_context import get_project_root

        token = str(identity or "").strip()
        if not token:
            return ""
        try:
            from infernux.core.assets import AssetManager
            from infernux.lib import ResourceType

            database = getattr(AssetManager, "_asset_database", None)
            if database is not None:
                path = str(
                    database.get_runtime_artifact_path(token, ResourceType.ParticleGraph) or ""
                )
                if path:
                    return path
        except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
            pass
        project_root = get_project_root()
        if not project_root:
            return ""
        return os.path.join(
            project_root,
            "Library",
            "Artifacts",
            "Particle",
            particle_artifact_filename(token),
        )

    @staticmethod
    def _load_persisted(
        artifact_path: str,
        *,
        key: str,
        source_hash: str,
        source_kind: str,
    ) -> ParticleArtifact | None:
        try:
            payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            if (
                type(payload) is not dict
                or set(payload) != {
                    "$schema", "source_key", "source_hash", "source_kind", "revision",
                    "semantic_hash", "behavior_hash", "hir", "kernel_ir", "gpu_glsl", "gpu_spirv",
                }
                or payload.get("$schema") != PARTICLE_ARTIFACT_SCHEMA
                or type(payload.get("source_key")) is not str
                or payload.get("source_hash") != source_hash
                or payload.get("source_kind") != source_kind
                or type(payload.get("hir")) is not dict
                or type(payload.get("kernel_ir")) is not dict
                or type(payload.get("gpu_glsl")) is not dict
                or type(payload.get("gpu_spirv")) is not dict
            ):
                return None
            revision = payload.get("revision")
            if type(revision) is not int or revision <= 0:
                return None
            runtime_metadata = decode_particle_runtime_metadata(payload["hir"])
            if runtime_metadata.behavior_hash != payload["behavior_hash"]:
                return None
            kernel_program = ParticleKernelProgram.from_dict(payload["kernel_ir"])
            if kernel_program.source_behavior_hash != payload["behavior_hash"]:
                return None
            gpu_program = GpuParticleGlslLowerer().lower(kernel_program)
            gpu_glsl = gpu_program.to_dict()
            if payload["gpu_glsl"] != gpu_glsl:
                return None
            gpu_spirv = validate_gpu_particle_spirv(payload["gpu_spirv"], gpu_program)
            return ParticleArtifact(
                key,
                source_hash,
                source_kind,
                revision,
                str(payload["semantic_hash"]),
                str(payload["behavior_hash"]),
                artifact_path,
                payload["hir"],
                kernel_program.to_dict(),
                gpu_glsl,
                gpu_spirv,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
            return None


def _program_to_dict(program: ParticleProgramHIR) -> dict[str, Any]:
    def operand(value):
        return {
            "type": value.value_type.to_dict(),
            "value_id": value.value_id,
            "literal": value.literal,
        }

    def stage(value):
        return {
            "stage": value.stage.value,
            "flow_id": value.flow_id,
            "root_uid": value.root_uid,
            "expressions": [
                {
                    "result_id": instruction.result_id,
                    "opcode": instruction.opcode,
                    "result_type": instruction.result_type.to_dict(),
                    "operands": [operand(item) for item in instruction.operands],
                    "immediates": list(instruction.immediates),
                    "source_node_uid": instruction.source_node_uid,
                    "source_port_id": instruction.source_port_id,
                }
                for instruction in value.expressions.instructions
            ],
            "flow": {
                "entry_node_uid": value.flow.entry_node_uid,
                "blocks": [
                    {
                        "node_uid": block.node_uid,
                        "operations": [
                            {
                                "opcode": operation.opcode,
                                "parameters": list(operation.parameters),
                                "source_node_uid": operation.source_node_uid,
                                "value_bindings": list(operation.value_bindings),
                                "execution_predicates": [
                                    {
                                        "source_node_uid": predicate.source_node_uid,
                                        "value_id": predicate.value_id,
                                        "literal": predicate.literal,
                                        "expected": predicate.expected,
                                        "runtime_condition": predicate.runtime_condition,
                                    }
                                    for predicate in operation.execution_predicates
                                ],
                            }
                            for operation in block.operations
                        ],
                        "incoming_edges": list(block.incoming_edges),
                        "outgoing_edges": list(block.outgoing_edges),
                    }
                    for block in value.flow.blocks
                ],
                "edges": [
                    {
                        "link_uid": edge.link_uid,
                        "source_node_uid": edge.source_node_uid,
                        "source_port_id": edge.source_port_id,
                        "target_node_uid": edge.target_node_uid,
                        "target_port_id": edge.target_port_id,
                        "predicate_node_uid": edge.predicate_node_uid,
                        "predicate_expected": edge.predicate_expected,
                        "lane_index": edge.lane_index,
                    }
                    for edge in value.flow.edges
                ],
                "lanes": [
                    {
                        "stable_id": lane.stable_id,
                        "index": lane.index,
                        "parent_index": lane.parent_index,
                        "source_node_uid": lane.source_node_uid,
                        "source_port_id": lane.source_port_id,
                    }
                    for lane in value.flow.lanes
                ],
                "joins": [
                    {
                        "node_uid": join.node_uid,
                        "input_lane_indices": list(join.input_lane_indices),
                        "output_lane_index": join.output_lane_index,
                    }
                    for join in value.flow.joins
                ],
                "suspensions": [
                    {
                        "node_uid": suspension.node_uid,
                        "kind": suspension.kind.value,
                        "lane_index": suspension.lane_index,
                        "lane_stable_id": suspension.lane_stable_id,
                        "resume_program_counter": suspension.resume_program_counter,
                        "resume_node_uid": suspension.resume_node_uid,
                        "value_id": suspension.value_id,
                        "literal": suspension.literal,
                    }
                    for suspension in value.flow.suspensions
                ],
            },
        }

    return {
        "stable_id": program.stable_id,
        "name": program.name,
        "semantic_hash": program.semantic_hash,
        "behavior_hash": program.behavior_hash,
        "parameters": [
            {
                "stable_id": parameter.stable_id,
                "name": parameter.name,
                "type": parameter.value_type.to_dict(),
                "default": parameter.default,
                "exposed": parameter.exposed,
                "writable": parameter.writable,
                "slot": parameter.slot,
                "category": parameter.category,
                "tooltip": parameter.tooltip,
                "hdr": bool(parameter.hdr),
            }
            for parameter in program.parameters
        ],
        "schedule": list(program.schedule.emitter_ids),
        "events": {
            "event_abi_hash": program.events.event_abi_hash,
            "event_types": [
                {
                    "stable_id": event_type.stable_id,
                    "name": event_type.name,
                    "type_index": event_type.type_index,
                    "stable_type_hash": event_type.stable_type_hash,
                    "queue_capacity": event_type.queue_capacity,
                    "payload_stride_words": event_type.payload_stride_words,
                    "fields": [
                        {
                            "stable_id": field.stable_id,
                            "name": field.name,
                            "type": field.value_type.to_dict(),
                            "word_offset": field.word_offset,
                            "word_count": field.word_count,
                            "default": field.default,
                        }
                        for field in event_type.fields
                    ],
                }
                for event_type in program.events.event_types
            ],
        },
        "emitters": [
            {
                "stable_id": emitter.stable_id,
                "name": emitter.name,
                "settings": emitter.settings.to_dict(),
                "attributes": [attribute.to_dict() for attribute in emitter.attributes],
                "data_interfaces": [
                    interface.to_dict() for interface in emitter.data_interfaces
                ],
                "init": stage(emitter.init),
                "update": stage(emitter.update),
                "collision_enter": (
                    stage(emitter.collision_enter)
                    if emitter.collision_enter is not None
                    else None
                ),
                "collision_stay": (
                    stage(emitter.collision_stay)
                    if emitter.collision_stay is not None
                    else None
                ),
                "collision_exit": (
                    stage(emitter.collision_exit)
                    if emitter.collision_exit is not None
                    else None
                ),
                "events": [stage(event_flow) for event_flow in emitter.event_flows],
                "rendering": stage(emitter.rendering),
                "render_plan": [
                    {
                        "output_id": output.output_id,
                        "output_type": output.output_type,
                        "mesh": output.mesh.to_dict(),
                        "mesh_parameter": output.mesh_parameter,
                        "shader": output.shader,
                        "shader_properties": [
                            {
                                "name": item.name,
                                "type": item.value_type.to_dict(),
                                "default": item.default,
                                "parameter_id": item.parameter_id,
                            }
                            for item in output.shader_properties
                        ],
                        "receive_scene_lighting": output.receive_scene_lighting,
                        "receive_shadows": output.receive_shadows,
                        "cast_shadows": output.cast_shadows,
                        "soft_particles": output.soft_particles,
                        "soft_distance": output.soft_distance,
                        "sort_mode": output.sort_mode,
                        "ribbon_uv_mode": output.ribbon_uv_mode,
                        "ribbon_uv_scale": output.ribbon_uv_scale,
                        "flipbook_columns": output.flipbook_columns,
                        "flipbook_rows": output.flipbook_rows,
                        "sprite_alignment": output.sprite_alignment,
                        "alignment_axis": list(output.alignment_axis),
                    }
                    for output in emitter.render_plan.outputs
                ],
            }
            for emitter in program.emitters
        ],
    }


__all__ = [
    "PARTICLE_ARTIFACT_SCHEMA",
    "ParticleArtifact",
    "ParticleArtifactError",
    "ParticleArtifactSuperseded",
    "ParticleArtifactRegistry",
    "ParticleGraphWriteTicket",
    "PreparedParticleGraphArtifact",
    "particle_artifact_filename",
]
