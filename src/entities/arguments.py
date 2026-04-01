#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser


class GroupParams:
    """Simple namespace object used to hold extracted parameter values."""


class ParamGroup:
    """Base class for argument groups that register their fields with argparse.

    Subclasses declare parameters as instance attributes in ``__init__`` before
    calling ``super().__init__``.  Attribute names that begin with an underscore
    get a single-character shorthand flag (e.g. ``_debug`` → ``--debug / -d``).
    """

    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        """Register all subclass attributes as argparse arguments.

        Args:
            parser: The ``ArgumentParser`` to add arguments to.
            name: Label for the argument group shown in ``--help`` output.
            fill_none: If ``True``, all default values are replaced with
                ``None``, allowing callers to detect unset arguments.
        """
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        """Copy matching arguments from a parsed namespace into a GroupParams.

        Args:
            args: Parsed ``argparse.Namespace`` returned by
                ``parser.parse_args()``.

        Returns:
            A ``GroupParams`` instance containing only the keys that belong to
            this group.
        """
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class OptimizationParams(ParamGroup):
    """Gaussian splatting optimisation hyperparameters.

    All fields are registered as CLI arguments via ``ParamGroup``.

    Attributes:
        iterations: Total number of optimisation iterations.
        position_lr_init: Initial learning rate for Gaussian positions.
        position_lr_final: Final learning rate for Gaussian positions after
            exponential decay.
        position_lr_delay_mult: Delay multiplier for the position LR schedule.
        position_lr_max_steps: Step count over which position LR decays.
        feature_lr: Learning rate for spherical-harmonic colour features.
        opacity_lr: Learning rate for per-Gaussian opacity values.
        scaling_lr: Learning rate for Gaussian scale parameters.
        rotation_lr: Learning rate for Gaussian rotation quaternions.
        percent_dense: Fraction of scene extent used as the densification
            size threshold.
        lambda_dssim: Weight of the D-SSIM term in the colour loss
            (``1 - lambda_dssim`` weights the L1 term).
        densification_interval: Iterations between densify-and-prune steps.
        opacity_reset_interval: Iterations between opacity resets.
        densify_from_iter: Iteration at which densification begins.
        densify_until_iter: Iteration at which densification stops.
        densify_grad_threshold: Gradient magnitude threshold above which a
            Gaussian is split or cloned.
    """

    def __init__(self, parser):
        """Initialise default values and register with the given parser.

        Args:
            parser: An ``ArgumentParser`` instance to register arguments on.
        """
        self.iterations = 30000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")
