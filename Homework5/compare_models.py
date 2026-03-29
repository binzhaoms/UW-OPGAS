import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    results_dir = Path("Homework5/results")
    fnn_path = results_dir / "fnn_results.json"
    cnn_path = results_dir / "cnn_results.json"

    if not fnn_path.exists() or not cnn_path.exists():
        print("Missing result files.")
        print("Run question1_fnn.py and question2_cnn.py first.")
        return

    fnn = load_json(fnn_path)
    cnn = load_json(cnn_path)

    fnn_acc = fnn["best"]["test_acc"]
    cnn_acc = cnn["best"]["test_acc"]

    comparison = {
        "fnn_best_test_acc": fnn_acc,
        "cnn_best_test_acc": cnn_acc,
        "delta_cnn_minus_fnn": cnn_acc - fnn_acc,
        "fnn_best_config": fnn["best"]["config"],
        "cnn_best_config": cnn["best"]["config"],
    }

    out_path = results_dir / "model_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Comparison complete:")
    print(json.dumps(comparison, indent=2))
    print("Saved: Homework5/results/model_comparison.json")


if __name__ == "__main__":
    main()
