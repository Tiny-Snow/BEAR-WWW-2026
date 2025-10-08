import os
import json
import argparse
import pandas as pd


def collect_results(checkpoint_dir: str, cbs_method: str):
    results: list[dict[str, float]] = []
    checkpoint_dirs = sorted(
        [
            int(d.split("-")[1])
            for d in os.listdir(checkpoint_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
    )
    for idx, checkpoint in enumerate(checkpoint_dirs, start=1):
        json_files = [
            f
            for f in os.listdir(
                os.path.join(checkpoint_dir, f"checkpoint-{checkpoint}")
            )
            if f.startswith("final_result_") and f.endswith(f"_{cbs_method}_match.json")
        ]
        if not json_files:
            print(f"No result file found for checkpoint-{checkpoint}")
            continue
        json_file = json_files[0]
        json_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint}", json_file)
        with open(json_path, "r") as f:
            result = json.load(f)
        rounded_result = {k: round(v, 4) for k, v in result.items()}
        rounded_result["epoch"] = idx
        results.append(rounded_result)

    df = pd.DataFrame(results)
    df = df[["epoch"] + [col for col in df.columns if col != "epoch"]]
    df.loc[len(df)] = ["max"] + [df[col].max() for col in df.columns[1:]]
    output_path = os.path.join(checkpoint_dir, f"results_{cbs_method}.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and save evaluation results from checkpoints."
    )
    parser.add_argument(
        "checkpoint_dir", type=str, help="Directory containing checkpoint folders."
    )
    parser.add_argument(
        "cbs_method",
        type=str,
        choices=["CBS", "customCBS"],
        help="CBS method used for evaluation.",
    )

    args = parser.parse_args()
    collect_results(args.checkpoint_dir, args.cbs_method)


if __name__ == "__main__":
    main()
