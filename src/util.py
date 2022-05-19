import argparse
import yaml


def create_argparser(default_yml_path: str) -> argparse.ArgumentParser:
    """Creates an argparser from a yaml file. Where each component is a flag to set and
    each subcomponent is an argument to be passed to the argparse
    .. code::
        epochs:
            help: Number of epochs to train over
            default: 100
    to make the code more flexible, help also be called desc default also called value.
    to indicate that this value both can be use to set a value and used to set a
    default. This also makes them compatible with the wieght and biases format:
    https://docs.wandb.ai/guides/track/config#file-based-configs
    Args:
        default_yml_path (str): Path to yaml file containing default arguments to parse.
    Returns:
        argparse.ArgumentParser: The ArgumentParser for parsing all the arguments in the
            config.
    """

    with open(default_yml_path, "r") as f:
        default_dict = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    for k, args in default_dict.items():
        if isinstance(args, dict):
            if ("value" in args) and ("default" in args):
                raise ValueError(
                    f"{k} contains both a 'value' and a 'default'," + " Remove one."
                )
            if ("desc" in args) and ("help" in args):
                raise ValueError(
                    f"{k} contains both a 'desc' and a 'help'." + " Remove one."
                )
            if "value" in args:
                args["default"] = args["value"].pop()
            if "desc" in args:
                args["default"] = args["value"].pop()
            parser.add_argument(f"--{k}", **args)

        # if not dict assume args is the default values
        else:
            parser.add_argument(f"--{k}", default=args)

    return parser
