import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

def load_data():
    feat_df = pd.read_csv("data/processed/dataset_features.csv", encoding="utf-8-sig")
    text_df = pd.read_csv("data/processed/dataset_text.csv", encoding="utf-8-sig")
    return feat_df, text_df

def train_svm(feat_df, text_df):
    print("[INFO] Mempersiapkan data...")

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(feat_df["author"])

    # Fitur stylometry
    X_style = feat_df.drop(columns=["author"]).values
    scaler = StandardScaler()
    X_style = scaler.fit_transform(X_style)

    # Fitur TF-IDF karakter n-gram
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
        sublinear_tf=True,
    )
    X_tfidf = tfidf.fit_transform(text_df["text_clean"])

    # Gabungkan fitur
    from scipy.sparse import csr_matrix
    X_combined = hstack([csr_matrix(X_style), X_tfidf])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[INFO] Training SVM...")
    model = SVC(kernel="linear", C=1, probability=True)
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Confusion Matrix - SVM Authorship Identification")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()
    print("\n[SAVED] results/confusion_matrix.png")

    # Cross validation
    print("\n[INFO] Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        SVC(kernel="rbf", C=10, gamma="scale"),
        X_combined, y, cv=5, scoring="accuracy"
    )
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, le, scaler, tfidf

if __name__ == "__main__":
    feat_df, text_df = load_data()
    model, le, scaler, tfidf = train_svm(feat_df, text_df)