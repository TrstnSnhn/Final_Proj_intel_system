from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models.nlp_classifier import train_tfidf_logreg


def generate_seed_text_data(path: Path):
    rows = [
        {"description": "dark spots on tomato leaves", "disease_label": "Tomato___Early_blight"},
        {"description": "powdery white fungus on squash leaves", "disease_label": "Squash___Powdery_mildew"},
        {"description": "healthy green apple leaves", "disease_label": "Apple___healthy"},
    ]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["train", "prepare"], default="train")
    args = parser.parse_args()

    data_csv = Path("data/disease_descriptions.csv")
    treatments_json = Path("data/disease_treatments.json")

    if not data_csv.exists():
        generate_seed_text_data(data_csv)
    if not treatments_json.exists():
        treatments_json.write_text(json.dumps({"Tomato___Early_blight": {"treatment": "Use fungicide"}}, indent=2))

    if args.action == "prepare":
        print("Prepared NLP seed data")
        return

    df = pd.read_csv(data_csv)
    x_train, x_test, y_train, y_test = train_test_split(
        df["description"].tolist(), df["disease_label"].tolist(), test_size=0.2, random_state=42
    )
    model = train_tfidf_logreg(x_train, y_train)
    preds = model.classifier.predict(model.vectorizer.transform(x_test))
    f1 = f1_score(y_test, preds, average="macro")

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "nlp_metrics.json").write_text(json.dumps({"macro_f1": float(f1)}, indent=2))
    print(f"NLP macro-F1: {f1:.4f}")


if __name__ == "__main__":
    main()
