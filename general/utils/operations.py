def get_module_from_name(lm_model, name):
    splits = name.split(".")
    module = lm_model
    for split in splits:
        if split == "":
            continue

        new_module = getattr(module, split)
        if new_module is None:
            raise ValueError(f"{module} has no attribute {split}.")
        module = new_module
    return module
