def main() -> None:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda: {torch.version.cuda}")  # pyright: ignore[reportAttributeAccessIssue]
        print(f"device: {torch.cuda.get_device_name(0)}")

    mps_available = torch.backends.mps.is_available()
    print(f"mps available: {mps_available}")


if __name__ == "__main__":
    main()
