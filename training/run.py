import argparse


def run_networks():
    from networks.train import main as run
    run()


def run_malwares():
    from malwares.train import main as run
    run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["networks", "malwares", "both"],
        default="networks"
    )

    args = parser.parse_args()
    print(f"Mode selected: {args.mode}")

    if args.mode == "networks":
        print("Running NETWORKS training...")
        run_networks()

    elif args.mode == "malwares":
        print("Running MALWARES training...")
        run_malwares()

    elif args.mode == "both":
        print("Running BOTH trainings...")
        run_networks()
        run_malwares()


if __name__ == "__main__":
    main()