from __future__ import annotations

import argparse

from .stages import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="D-HASH Experiment Runner (Modular Edition)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "pipeline", "microbench", "ablation", "zipf", "redistrib"],
        default="all",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Zipf alpha for Ablation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["NASA", "eBay", "ALL"],
        default="ALL",
    )
    parser.add_argument(
        "--fixed_window",
        type=int,
        default=None,
        help="Fixed W for D-HASH (defaults to chosen pipeline size or 500)",
    )
    parser.add_argument(
        "--dhash_T",
        type=int,
        default=None,
        help="D-HASH threshold T for zipf main (default: max(30, W))",
    )
    parser.add_argument(
        "--pipeline_for_zipf",
        type=int,
        default=None,
        help="Pipeline size B to use in main Zipf (defaults to fixed_window or 500)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats for all stages",
    )
    parser.add_argument(
        "--algos",
        type=str,
        choices=["auto", "minimal", "all", "custom"],
        default="auto",
        help="Algorithm set selector",
    )
    parser.add_argument(
        "--algos_list",
        type=str,
        default="",
        help="Comma list for --algos custom. Choose from: ch, wch, hrw, dhash",
    )

    args = parser.parse_args()

    run_experiments(
        mode=args.mode,
        alpha_for_ablation=args.alpha,
        dataset_filter=args.dataset,
        fixed_window=args.fixed_window,
        dhash_T=args.dhash_T,
        pipeline_for_zipf=args.pipeline_for_zipf,
        repeats=args.repeats,
        algos=args.algos,
        algos_list=args.algos_list,
    )


if __name__ == "__main__":
    main()
