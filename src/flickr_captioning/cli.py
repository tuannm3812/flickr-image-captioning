from __future__ import annotations

import argparse

from flickr_captioning.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flickr-caption")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an image captioning model.")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--model", choices=["baseline", "attention"], default="baseline")

    predict_parser = subparsers.add_parser("predict", help="Generate a caption for one image.")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--max-length", type=int, default=None)
    predict_parser.add_argument("--beam-size", type=int, default=1)
    predict_parser.add_argument("--model", choices=["baseline", "attention"], default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate BLEU scores on the test split.")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--beam-size", type=int, default=1)
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--model", choices=["baseline", "attention"], default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        from flickr_captioning.train import train

        config = load_config(args.config)
        checkpoint = train(config, model_kind=args.model)
        print(f"Saved best checkpoint to {checkpoint}")
    elif args.command == "predict":
        from flickr_captioning.inference import predict

        caption = predict(
            args.checkpoint,
            args.image,
            device_name=args.device,
            max_length=args.max_length,
            beam_size=args.beam_size,
        )
        print(caption)
    elif args.command == "evaluate":
        from flickr_captioning.evaluation import evaluate_bleu

        scores = evaluate_bleu(
            args.checkpoint,
            device_name=args.device,
            beam_size=args.beam_size,
            limit=args.limit,
        )
        for metric, value in scores.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
