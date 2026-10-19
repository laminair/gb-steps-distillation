"""render_gold_config.validate() must check per-arm lmbda/beta requirements over a whole schedule."""

import pytest
import yaml

import render_gold_config as rgc

BASE = [
    "--student-model-path", "/s", "--teacher-model-path", "/t", "--teacher-tokenizer-path", "/tt",
    "--corpus-path", "/c", "--output-dir", "/o", "--deepspeed-config", "/d", "--out", "/x",
    "--nodes", "2", "--gpus-per-node", "8",
]


def _args(lmbda, loss_arm="jsd", extra=None, *more):
    argv = BASE + ["--lmbda", str(lmbda), "--loss-arm", loss_arm, *more]
    if extra:
        argv += ["--extra-config-yaml", yaml.safe_dump(extra)]
    return rgc.build_parser().parse_args(argv)


def _validate(*a, **kw):
    rgc.validate(_args(*a, **kw), check_paths=False)


def test_constant_configs_still_pass():
    _validate(1.0, "sampled_opd", None, "--last-message-only")
    _validate(0.5, "distillm2")
    _validate(0.0, "ce", None, "--vllm-num-servers", "0")


def test_sampled_opd_rejects_ramp_from_off_policy():
    with pytest.raises(rgc.ConfigError, match="whole run"):
        _validate(1.0, "sampled_opd", {"lmbda_schedule": "linear", "lmbda_init": 0.0},
                  "--last-message-only")


def test_sampled_opd_accepts_flat_linear_schedule():
    _validate(1.0, "sampled_opd", {"lmbda_schedule": "linear", "lmbda_init": 1.0},
              "--last-message-only")


def test_distillm2_rejects_ramp_touching_zero():
    with pytest.raises(rgc.ConfigError, match=r"0\.0 < lmbda < 1\.0"):
        _validate(0.5, "distillm2", {"lmbda_schedule": "linear", "lmbda_init": 0.0})


def test_distillm2_accepts_ramp_strictly_inside():
    _validate(0.7, "distillm2", {"lmbda_schedule": "linear", "lmbda_init": 0.2})


def test_jsd_accepts_full_ramp():
    # The case the trainer fix was for: no arm requirement, ramp from 0.0 to 1.0.
    _validate(1.0, "jsd", {"lmbda_schedule": "linear", "lmbda_init": 0.0})


def test_downward_ramp_to_off_policy_rejected():
    with pytest.raises(rgc.ConfigError, match="ramps DOWN"):
        _validate(0.0, "jsd", {"lmbda_schedule": "linear", "lmbda_init": 1.0},
                  "--vllm-num-servers", "0")


def test_beta_requirement_checks_whole_ramp():
    with pytest.raises(rgc.ConfigError, match="beta ranging"):
        rgc.validate(
            _args(0.5, "liger_fused_kl_interpolation",
                  {"beta_schedule": "linear", "beta_init": 0.0}),
            check_paths=False,
        )


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"lmbda_schedule": "linear"}, "requires lmbda_init"),
        ({"lmbda_schedule": "linear", "lmbda_init": 1.5}, r"in \[0, 1\]"),
        ({"lmbda_schedule": "linear", "lmbda_init": "abc"}, "must be a number"),
        ({"lmbda_schedule": "cosine"}, "not one of"),
    ],
)
def test_bad_schedule_rejected_at_render_time(extra, match):
    with pytest.raises(rgc.ConfigError, match=match):
        _validate(0.5, "jsd", extra)
