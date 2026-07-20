def main() -> None:
    import torch

    print(torch.__version__)
    print(torch.version.cuda)  # pyright: ignore[reportAttributeAccessIssue]

    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
