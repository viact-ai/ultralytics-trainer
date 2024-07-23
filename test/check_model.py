from clearml import Model


def main():
    model = Model(model_id="dcdfc64507bb4b108fece3aff9803f50")
    print("Model id:", model.id)
    print("Model URL:", model.url)


if __name__ == "__main__":
    main()
