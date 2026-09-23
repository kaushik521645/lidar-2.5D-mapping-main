"""
FoveaMap One-Command Demo Launcher.

Usage:
  python run_demo.py                  # Launches server with continuous auto-looping across all sequences
  python run_demo.py -s 04            # Launches server starting with KITTI sequence 04
  python run_demo.py --no-auto-loop   # Replays only the single specified sequence in loop
  python run_demo.py --all-sequences  # Precomputes frames for all 22 sequences in dataset
  python run_demo.py --eval           # Runs evaluation and benchmarks report
  python run_demo.py --export         # Recomputes and exports scenario JSONs
"""

import os
import sys
import argparse
import webbrowser
import uvicorn
import yaml

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.kitti_loader import KITTILoader
from src.ingestion.extract import ensure_velodyne_extracted
from src.ingestion.sample_generator import generate_kitti_sample_dataset
from src.api.export_precomputed import export_all_precomputed, export_kitti_sequence, export_synthetic_kitti_like
from src.metrics.evaluator import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="FoveaMap Perception Dashboard Runner")
    parser.add_argument("--sequence", type=str, default="00", help="KITTI Sequence to load (e.g. 00, 01, ..., 21)")
    parser.add_argument("--no-auto-loop", action="store_true", help="Disable automatic sequence looping")
    parser.add_argument("--all-sequences", action="store_true", help="Precompute frames for all available sequences in dataset")
    parser.add_argument("--eval", action="store_true", help="Run full evaluation and print report")
    parser.add_argument("--export", action="store_true", help="Re-export precomputed scenario JSONs")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    active_seq = str(args.sequence).zfill(2)

    extract_dir, did_extract = ensure_velodyne_extracted()
    if extract_dir:
        print(
            f"[INFO] Velodyne data dir : {extract_dir}"
            f"{' (extracted this run)' if did_extract else ' (on-disk, zip ignored if present)'}"
        )

    # Step 1: Detect available datasets (Real KITTI Odometry Velodyne alongside SemanticKITTI Benchmark)
    kitti_loader = KITTILoader(sequence=active_seq)
    available_seqs = kitti_loader.list_sequences()

    if len(kitti_loader) > 0 and not kitti_loader.is_synthetic_signature:
        print(f"[INFO] Primary Dataset  : Genuine KITTI Odometry Velodyne ({os.path.basename(kitti_loader.dataset_path)})")
        print(f"[INFO] Available Seqs   : {', '.join(available_seqs)} ({len(available_seqs)} sequences)")
        print(f"[INFO] Selected Sequence: Sequence {active_seq} ({len(kitti_loader)} scans)")

    kitti_dir = "data/synthetic_kitti_like"
    if not os.path.exists(os.path.join(kitti_dir, "velodyne")):
        print("[INFO] Generating SemanticKITTI benchmark sequence in data/synthetic_kitti_like...")
        generate_kitti_sample_dataset(kitti_dir, num_frames=30)
    print(f"[INFO] SemanticKITTI    : Ground-truth labeled benchmark available in {kitti_dir}")

    # Step 2: Handle --eval flag
    if args.eval:
        run_full_evaluation()
        return

    # Step 3: Handle --export or --all-sequences flag
    if args.all_sequences:
        print(f"[INFO] Exporting all {len(available_seqs)} KITTI sequences...")
        export_all_precomputed(export_all_sequences=True)
        return

    if args.export:
        print(f"[INFO] Exporting precomputed frames for sequence {active_seq} and SemanticKITTI...")
        export_all_precomputed(sequence=active_seq)
        return

    # Ensure precomputed files exist for the selected sequence
    precomputed_target = f"data/precomputed/kitti_seq_{active_seq}.json"
    if active_seq == "00" and os.path.exists("data/precomputed/kitti_odometry.json"):
        precomputed_target = "data/precomputed/kitti_odometry.json"

    if not os.path.exists(precomputed_target):
        print(f"[INFO] Precomputing initial frames for Sequence {active_seq}...")
        export_kitti_sequence(sequence=active_seq)

    # Ensure SemanticKITTI benchmark is precomputed alongside
    synth_target = "data/precomputed/synthetic_kitti_like.json"
    if not os.path.exists(synth_target) or os.path.getsize(synth_target) > 100 * 1024 * 1024:
        print(f"[INFO] Precomputing SemanticKITTI benchmark frames in {synth_target}...")
        export_synthetic_kitti_like()

    # Step 4: Launch Web Server & Dashboard
    dashboard_url = f"http://localhost:{args.port}/"
    print("\n" + "=" * 70)
    print("      FOVEAMAP PERCEPTION DASHBOARD & PIPELINE RUNNING")
    print(f"      Dashboard URL   : {dashboard_url}")
    print(f"      WebSocket API   : ws://localhost:{args.port}/ws/grid")
    print(f"      REST Health     : http://localhost:{args.port}/health")
    print(f"      Active Dataset  : {os.path.basename(kitti_loader.dataset_path)}")
    auto_loop_mode = "DISABLED (Single Sequence Repeat)" if args.no_auto_loop else "ENABLED (Continuous Playlist across all sequences)"
    print(f"      Active Sequence : Sequence {active_seq} ({len(kitti_loader)} scans)")
    print(f"      Auto-Loop Mode  : {auto_loop_mode}")
    print(f"      Sequences Loaded: {len(available_seqs)} available ({available_seqs[0]} to {available_seqs[-1]})")
    print("=" * 70 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(dashboard_url)
        except Exception as e:
            print(f"[WARN] Could not auto-open browser: {e}. Please open {dashboard_url} manually.")

    from src.api.server import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
