import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        help="The datset name, e.g., Toy",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=["lml", "msl", "bear"],
        help="loss type",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the pre-trained LLM, e.g., '/path/to/Llama-3.2-3B'",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        required=True,
        help='The CUDA_VISIBLE_DEVICES value, e.g., "0,1,2,3"',
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="The number of epochs, e.g., 5",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10000,
        help="The sample size, default is 10000",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="The temperature parameter for MSL, default is 1.0",
    )
    parser.add_argument(
        "--topk_tau",
        type=float,
        default=1.0,
        help="The temperature parameter for Top-K loss in MSL@K, default is 1.0",
    )
    parser.add_argument(
        "--topk_weight",
        type=float,
        default=1.0,
        help="The weight for topk loss, default is 1.0",
    )
    parser.add_argument("--no_training", action="store_true", help="If set, skip the training step.")
    parser.add_argument("--no_inference", action="store_true", help="If set, skip the inference step.")

    args = parser.parse_args()
    model_name = os.path.basename(args.model_path)
    print(f">>>>> Using Base LLM: {model_name}")

    # 1. cd to the code directory
    code_dir = "/path/to/your/code"     # TODO: change to your code path
    os.chdir(code_dir)

    # 2. run train.py
    if not args.no_training:
        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = args.cuda
        train_cmd = [
            "accelerate",
            "launch",
            f"train_{args.loss}.py",
            "--base_model",
            args.model_path,
            "--dataset_name",
            args.dataset,
            "--sample",
            str(args.sample),
            "--num_epochs",
            str(args.num_epochs),
            "--tau",
            str(args.tau),
        ]
        if args.loss in ["bear"]:
            train_cmd += [
                "--topk_tau",
                str(args.topk_tau),
                "--topk_weight",
                str(args.topk_weight),
            ]
        print(f">>>>> Running training:")
        print(f">>>>> {' '.join(train_cmd)}")
        subprocess.run(train_cmd, env=train_env, check=True)
        print(">>>>> Training completed.")

    # 3. run inference.py
    loss_info = f"sample{args.sample}_epoch{args.num_epochs}_{args.loss}_tau{args.tau}"
    if args.loss in ["bear"]:
        loss_info += f"_topktau{args.topk_tau}"
        loss_info += f"_topkw{args.topk_weight}"
    save_dir = os.path.join(
        code_dir,
        "save_lora_model",
        model_name,
        args.dataset,
        loss_info,
    )
    # find the latest run, e.g., save_dir/1, then let save_dir = save_dir/1
    if os.path.exists(save_dir):
        subdirs = [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]
        if subdirs:
            latest_run = max(subdirs, key=lambda x: int(x))
            save_dir = os.path.join(save_dir, latest_run)
    else:
        raise FileNotFoundError(f"Directory {save_dir} does not exist.")
    print(f">>>>> Using save directory: {save_dir}")
    # find all the checkpoint directories, e.g., save_dir/checkpoint-417
    checkpoint_dirs = [
        d for d in os.listdir(save_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(save_dir, d))
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {save_dir}.")
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
    if not args.no_inference:
        for epoch, checkpoint in enumerate(checkpoint_dirs, start=1):
            checkpoint_path = os.path.join(save_dir, checkpoint)
            print(f">>>>> Running inference for {checkpoint_path} (epoch {epoch})")
            inference_env = os.environ.copy()
            inference_env["CUDA_VISIBLE_DEVICES"] = args.cuda
            inference_cmd = [
                "accelerate",
                "launch",
                "inference.py",
                "--base_model",
                args.model_path,
                "--lora_weights_path",
                checkpoint_path,
                "--dataset",
                args.dataset,
                "--constrained_before_softmax",
                "1" if args.loss in ["msl", "bear"] else "0",
            ]
            print(f">>>>> {' '.join(inference_cmd)}")
            subprocess.run(inference_cmd, env=inference_env, check=True)
        print(">>>>> Inference completed.")

    # 4. run evaluate_batch_match.py
    print(">>>>> Running evaluation...")
    eval_env = os.environ.copy()
    # only use one GPU for evaluation
    eval_env["CUDA_VISIBLE_DEVICES"] = args.cuda.split(",")[0]
    eval_cmd = [
        "python",
        "evaluate_batch_match.py",
        "--lora_weights_father_path",
        save_dir,
    ]
    print(f">>>>> {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, env=eval_env, check=True)

    # 5. run read_metric.py
    metric_cmd = [
        "python",
        "read_metric.py",
        save_dir,
        "customCBS" if args.loss in ["msl", "bear"] else "CBS",
    ]
    print(f">>>>> {' '.join(metric_cmd)}")
    subprocess.run(metric_cmd, env=eval_env, check=True)
    print(">>>>> Evaluation completed.")


if __name__ == "__main__":
    main()
