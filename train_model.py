from modeling import DATA_PATH, run_model_selection, load_data, save_training_artifacts


def main() -> None:
    data = load_data(DATA_PATH)
    artifacts = run_model_selection(data, DATA_PATH)
    save_training_artifacts(artifacts)
    print("Training complete. Artifacts written to ./artifacts")


if __name__ == "__main__":
    main()
