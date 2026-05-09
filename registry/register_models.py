from registry.model_registry import ModelVersionManager


def main():
    manager = ModelVersionManager()
    versions = manager.register_all_v1()
    for model_name, version in versions.items():
        print(f"{model_name}: registered version {version.version} as Staging")


if __name__ == "__main__":
    main()
