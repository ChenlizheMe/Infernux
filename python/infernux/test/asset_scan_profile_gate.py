"""Real 10k-file cold start, hot restart, and steady refresh gate."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from infernux import Engine
from infernux.lib import RuntimeMode


ASSET_COUNT = 10_000
COLD_WALL_LIMIT_SECONDS = 90.0
HOT_WALL_LIMIT_SECONDS = 8.0
HOT_SCAN_LIMIT_MS = 5_000.0
HOT_COMMIT_LIMIT_MS = 2_000.0
STEADY_COMMIT_LIMIT_MS = 500.0


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="infernux-asset-scan-profile-") as root:
        project = Path(root)
        assets = project / "Assets"
        fixture = assets / "scan-10k"
        fixture.mkdir(parents=True)
        (project / "ProjectSettings").mkdir()
        for index in range(ASSET_COUNT):
            (fixture / f"asset-{index:05d}.txt").write_text(str(index), encoding="ascii")

        started = time.perf_counter()
        cold_engine = Engine(mode=RuntimeMode.Headless)
        try:
            cold_engine.init_headless(str(project))
            cold_wall_seconds = time.perf_counter() - started
            cold_database = cold_engine.get_asset_database()
            if cold_database.last_refresh_scan_on_worker is not True:
                raise AssertionError("cold startup performed filesystem scan on the owner thread")
            if cold_database.last_refresh_scanned_count < ASSET_COUNT:
                raise AssertionError(
                    f"cold startup scanned only {cold_database.last_refresh_scanned_count} files"
                )
            if cold_database.last_refresh_imported_count < ASSET_COUNT:
                raise AssertionError(
                    f"cold startup imported only {cold_database.last_refresh_imported_count} files"
                )
            if cold_database.last_refresh_metadata_task_count < ASSET_COUNT:
                raise AssertionError(
                    "cold startup did not schedule metadata preparation for every new asset"
                )
            if (
                cold_database.last_refresh_worker_metadata_count
                != cold_database.last_refresh_metadata_task_count
            ):
                raise AssertionError("cold startup prepared metadata on the owner thread")
            cold_metrics = {
                "scan_ms": cold_database.last_refresh_scan_ms,
                "prepare_ms": cold_database.last_refresh_prepare_ms,
                "finalize_ms": cold_database.last_refresh_finalize_ms,
                "owner_merge_max_slice_ms": (
                    cold_database.last_refresh_owner_merge_max_slice_ms
                ),
                "owner_merge_slice_count": int(
                    cold_database.last_refresh_owner_merge_slice_count
                ),
                "metadata_write_ms": cold_database.last_refresh_metadata_write_ms,
                "journal_uncompressed_bytes": int(
                    cold_database.last_refresh_journal_uncompressed_bytes
                ),
                "journal_bytes": int(cold_database.last_refresh_journal_bytes),
                "journal_serialize_ms": cold_database.last_refresh_journal_serialize_ms,
                "journal_write_ms": cold_database.last_refresh_journal_write_ms,
                "journal_apply_ms": cold_database.last_refresh_journal_apply_ms,
                "import_ms": cold_database.last_refresh_import_ms,
                "index_build_ms": cold_database.last_refresh_index_build_ms,
                "index_save_ms": cold_database.last_refresh_index_save_ms,
                "index_build_on_worker": bool(
                    cold_database.last_refresh_index_build_on_worker
                ),
                "query_build_ms": cold_database.last_refresh_query_build_ms,
                "query_build_on_worker": bool(
                    cold_database.last_refresh_query_build_on_worker
                ),
                "dependency_build_ms": cold_database.last_refresh_dependency_build_ms,
                "dependency_build_on_worker": bool(
                    cold_database.last_refresh_dependency_build_on_worker
                ),
                "publish_ms": cold_database.last_refresh_publish_ms,
                "metadata_tasks": int(cold_database.last_refresh_metadata_task_count),
                "worker_metadata": int(cold_database.last_refresh_worker_metadata_count),
            }
            transaction_journal = Path(cold_database.asset_index_path).with_name(
                "AssetRefresh.transaction"
            )
            if transaction_journal.exists():
                raise AssertionError("cold startup left its metadata transaction journal behind")
            if not Path(cold_database.asset_index_path).is_file():
                raise AssertionError("cold startup did not rebuild AssetIndex after journal commit")
            if cold_metrics["journal_uncompressed_bytes"] <= 0:
                raise AssertionError("cold startup did not record a metadata transaction payload")
            if cold_metrics["journal_bytes"] >= cold_metrics["journal_uncompressed_bytes"]:
                raise AssertionError("metadata transaction journal was not compressed")
            if cold_metrics["journal_serialize_ms"] > 5000.0:
                raise AssertionError(
                    "metadata transaction journal serialization exceeded the owner-independent gate"
                )
            if cold_metrics["index_build_on_worker"] is not True:
                raise AssertionError("cold startup built AssetIndex on the owner thread")
            if cold_metrics["query_build_on_worker"] is not True:
                raise AssertionError("cold startup built its query snapshot on the owner thread")
            if cold_metrics["dependency_build_on_worker"] is not True:
                raise AssertionError("cold startup built its dependency snapshot on the owner thread")
            if cold_metrics["owner_merge_slice_count"] < 3:
                raise AssertionError("cold startup did not budget owner artifact merging")
            if cold_metrics["owner_merge_max_slice_ms"] > 50.0:
                raise AssertionError(
                    "cold startup exceeded the 50ms owner artifact merge slice gate"
                )
            if cold_wall_seconds > COLD_WALL_LIMIT_SECONDS:
                raise AssertionError(
                    f"10k cold startup exceeded {COLD_WALL_LIMIT_SECONDS:.0f}s: "
                    f"{cold_wall_seconds:.3f}s"
                )
        finally:
            cold_engine.exit()

        started = time.perf_counter()
        hot_engine = Engine(mode=RuntimeMode.Headless)
        try:
            hot_engine.init_headless(str(project))
            hot_wall_seconds = time.perf_counter() - started
            hot_database = hot_engine.get_asset_database()
            hot_metrics = {
                "scan_ms": hot_database.last_refresh_scan_ms,
                "commit_ms": hot_database.last_refresh_commit_ms,
                "prepare_ms": hot_database.last_refresh_prepare_ms,
                "finalize_ms": hot_database.last_refresh_finalize_ms,
                "owner_merge_max_slice_ms": (
                    hot_database.last_refresh_owner_merge_max_slice_ms
                ),
                "owner_merge_slice_count": int(
                    hot_database.last_refresh_owner_merge_slice_count
                ),
                "metadata_write_ms": hot_database.last_refresh_metadata_write_ms,
                "restore_ms": hot_database.last_refresh_restore_ms,
                "import_ms": hot_database.last_refresh_import_ms,
                "index_build_ms": hot_database.last_refresh_index_build_ms,
                "index_save_ms": hot_database.last_refresh_index_save_ms,
                "query_build_ms": hot_database.last_refresh_query_build_ms,
                "query_build_on_worker": bool(
                    hot_database.last_refresh_query_build_on_worker
                ),
                "dependency_build_ms": hot_database.last_refresh_dependency_build_ms,
                "dependency_build_on_worker": bool(
                    hot_database.last_refresh_dependency_build_on_worker
                ),
                "publish_ms": hot_database.last_refresh_publish_ms,
                "reused": int(hot_database.last_refresh_reused_count),
                "metadata_tasks": int(hot_database.last_refresh_metadata_task_count),
                "worker_metadata": int(hot_database.last_refresh_worker_metadata_count),
                "scan_on_worker": bool(hot_database.last_refresh_scan_on_worker),
            }
            if hot_metrics["scan_on_worker"] is not True:
                raise AssertionError("hot restart performed filesystem scan on the owner thread")
            if hot_metrics["reused"] < ASSET_COUNT:
                raise AssertionError(
                    f"hot restart reused only {hot_metrics['reused']} files"
                )
            if hot_database.last_refresh_imported_count != 0:
                raise AssertionError(
                    f"hot restart imported {hot_database.last_refresh_imported_count} unchanged assets"
                )
            if hot_metrics["index_build_ms"] != 0.0 or hot_metrics["index_save_ms"] != 0.0:
                raise AssertionError("hot restart rebuilt or rewrote an unchanged AssetIndex")
            if hot_metrics["query_build_on_worker"] is not True:
                raise AssertionError("hot restart built its query snapshot on the owner thread")
            if hot_metrics["dependency_build_on_worker"] is not True:
                raise AssertionError("hot restart built its dependency snapshot on the owner thread")
            if hot_metrics["scan_ms"] > HOT_SCAN_LIMIT_MS:
                raise AssertionError(
                    f"10k hot restart scan exceeded {HOT_SCAN_LIMIT_MS:.0f}ms: "
                    f"{hot_metrics['scan_ms']:.3f}ms"
                )
            if hot_metrics["commit_ms"] > HOT_COMMIT_LIMIT_MS:
                raise AssertionError(
                    f"10k hot restart commit exceeded {HOT_COMMIT_LIMIT_MS:.0f}ms: "
                    f"{hot_metrics['commit_ms']:.3f}ms"
                )
            if hot_wall_seconds > HOT_WALL_LIMIT_SECONDS:
                raise AssertionError(
                    f"10k hot restart exceeded {HOT_WALL_LIMIT_SECONDS:.0f}s: "
                    f"{hot_wall_seconds:.3f}s"
                )

            query_generation = hot_database.query_generation
            catalog_generation = hot_database.catalog_generation
            started = time.perf_counter()
            hot_database.refresh()
            steady_wall_seconds = time.perf_counter() - started
            if hot_database.query_generation != query_generation:
                raise AssertionError("unchanged steady refresh published a new query generation")
            if hot_database.catalog_generation != catalog_generation:
                raise AssertionError("unchanged steady refresh rebuilt the asset catalog")
            if hot_database.last_refresh_imported_count != 0:
                raise AssertionError("unchanged steady refresh ran importers")
            if any(
                value != 0.0
                for value in (
                    hot_database.last_refresh_restore_ms,
                    hot_database.last_refresh_import_ms,
                    hot_database.last_refresh_index_build_ms,
                    hot_database.last_refresh_index_save_ms,
                    hot_database.last_refresh_query_build_ms,
                    hot_database.last_refresh_dependency_build_ms,
                    hot_database.last_refresh_publish_ms,
                )
            ):
                raise AssertionError("unchanged steady refresh executed an owner commit phase")
            if hot_database.last_refresh_commit_ms > STEADY_COMMIT_LIMIT_MS:
                raise AssertionError(
                    f"10k steady no-op commit exceeded {STEADY_COMMIT_LIMIT_MS:.0f}ms: "
                    f"{hot_database.last_refresh_commit_ms:.3f}ms"
                )

            print(
                json.dumps(
                    {
                        "asset_count": ASSET_COUNT,
                        "cold_wall_seconds": round(cold_wall_seconds, 4),
                        "cold_scan_ms": round(cold_metrics["scan_ms"], 4),
                        "cold_prepare_ms": round(cold_metrics["prepare_ms"], 4),
                        "cold_finalize_ms": round(cold_metrics["finalize_ms"], 4),
                        "cold_owner_merge_max_slice_ms": round(
                            cold_metrics["owner_merge_max_slice_ms"], 4
                        ),
                        "cold_owner_merge_slice_count": cold_metrics[
                            "owner_merge_slice_count"
                        ],
                        "cold_metadata_write_ms": round(cold_metrics["metadata_write_ms"], 4),
                        "cold_journal_uncompressed_bytes": cold_metrics[
                            "journal_uncompressed_bytes"
                        ],
                        "cold_journal_bytes": cold_metrics["journal_bytes"],
                        "cold_journal_serialize_ms": round(
                            cold_metrics["journal_serialize_ms"], 4
                        ),
                        "cold_journal_write_ms": round(
                            cold_metrics["journal_write_ms"], 4
                        ),
                        "cold_journal_apply_ms": round(
                            cold_metrics["journal_apply_ms"], 4
                        ),
                        "cold_import_ms": round(cold_metrics["import_ms"], 4),
                        "cold_index_build_ms": round(cold_metrics["index_build_ms"], 4),
                        "cold_index_save_ms": round(cold_metrics["index_save_ms"], 4),
                        "cold_query_build_ms": round(cold_metrics["query_build_ms"], 4),
                        "cold_dependency_build_ms": round(
                            cold_metrics["dependency_build_ms"], 4
                        ),
                        "cold_index_build_on_worker": cold_metrics[
                            "index_build_on_worker"
                        ],
                        "cold_publish_ms": round(cold_metrics["publish_ms"], 4),
                        "cold_metadata_tasks": cold_metrics["metadata_tasks"],
                        "cold_worker_metadata": cold_metrics["worker_metadata"],
                        "hot_restart_wall_seconds": round(hot_wall_seconds, 4),
                        "hot_restart_scan_ms": round(hot_metrics["scan_ms"], 4),
                        "hot_restart_commit_ms": round(hot_metrics["commit_ms"], 4),
                        "hot_restart_prepare_ms": round(hot_metrics["prepare_ms"], 4),
                        "hot_restart_finalize_ms": round(hot_metrics["finalize_ms"], 4),
                        "hot_restart_owner_merge_max_slice_ms": round(
                            hot_metrics["owner_merge_max_slice_ms"], 4
                        ),
                        "hot_restart_owner_merge_slice_count": hot_metrics[
                            "owner_merge_slice_count"
                        ],
                        "hot_restart_metadata_write_ms": round(hot_metrics["metadata_write_ms"], 4),
                        "hot_restart_restore_ms": round(hot_metrics["restore_ms"], 4),
                        "hot_restart_import_ms": round(hot_metrics["import_ms"], 4),
                        "hot_restart_index_build_ms": round(hot_metrics["index_build_ms"], 4),
                        "hot_restart_index_save_ms": round(hot_metrics["index_save_ms"], 4),
                        "hot_restart_query_build_ms": round(hot_metrics["query_build_ms"], 4),
                        "hot_restart_dependency_build_ms": round(
                            hot_metrics["dependency_build_ms"], 4
                        ),
                        "hot_restart_publish_ms": round(hot_metrics["publish_ms"], 4),
                        "hot_restart_reused": hot_metrics["reused"],
                        "hot_restart_metadata_tasks": hot_metrics["metadata_tasks"],
                        "hot_restart_worker_metadata": hot_metrics["worker_metadata"],
                        "steady_wall_seconds": round(steady_wall_seconds, 4),
                        "steady_scan_ms": round(hot_database.last_refresh_scan_ms, 4),
                        "steady_commit_ms": round(hot_database.last_refresh_commit_ms, 4),
                        "scan_on_worker": hot_metrics["scan_on_worker"],
                    },
                    sort_keys=True,
                )
            )
        finally:
            hot_engine.exit()


if __name__ == "__main__":
    main()
