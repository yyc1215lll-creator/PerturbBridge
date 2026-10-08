"""
Train a diffusion model on images.
"""

import argparse
import copy

from ddbm import dist_util, logger
from datasets import load_data
from ddbm.resample import create_named_schedule_sampler
from ddbm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir,
)
from ddbm.train_util import TrainLoop
from ddbm.cellflux_fid import CellFluxFIDEvaluator
from ddbm.auxiliary_losses import (
    AuxiliaryLossManager,
    AuxiliaryLossSchedule,
    ConditionalMomentLoss,
    FIDInceptionV3Encoder,
    FID_INCEPTION_ENCODER,
    SupervisedPrototypeInfoNCELoss,
    load_condition_moa_labels,
    load_conditional_statistics,
    sha256_file,
)

import torch
import torch.distributed as dist

from pathlib import Path

from ddbm.tracking import wandb

from glob import glob
import os
from datasets.augment import AugmentPipe


def main(args):

    workdir = get_workdir(args.exp, args.workdir_root)
    Path(workdir).mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + "_resume"
        wandb.init(
            project="bridge",
            group=args.exp,
            name=name,
            config=vars(args),
            mode="offline" if not args.debug else "disabled",
        )
        logger.log("creating model and diffusion...")

    data_image_size = args.image_size

    # Load target model
    resume_train_flag = args.resume_checkpoint != ""
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f"{workdir}/*model*[0-9].*"))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split("model_")[-1].split(".")[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                resume_train_flag = True
        elif args.pretrained_ckpt is not None:
            max_ckpt = args.pretrained_ckpt
            args.resume_checkpoint = max_ckpt
        if dist.get_rank() == 0 and args.resume_checkpoint != "":
            logger.log("Resuming from checkpoint: ", max_ckpt)

    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())

    auxiliary_loss_schedules = {}
    conditional_statistics = None
    if args.moment_loss_enabled or args.infonce_loss_enabled:
        if args.dataset != "cellflux_bbbc021":
            raise ValueError("feature auxiliary losses are currently supported only for cellflux_bbbc021")
        if args.auxiliary_encoder != FID_INCEPTION_ENCODER:
            raise ValueError(f"unsupported auxiliary encoder: {args.auxiliary_encoder}")
        if args.moment_loss_enabled and args.cellflux_condition_group_size < args.moment_min_group_size:
            raise ValueError("condition group size is smaller than moment_min_group_size")
        conditional_statistics = load_conditional_statistics(
            args.moment_stats_path,
            expected_encoder=args.auxiliary_encoder,
            expected_weights_sha256=args.auxiliary_encoder_weights_sha256,
        )
        encoder = FIDInceptionV3Encoder(
            args.auxiliary_encoder_weights_path,
            expected_sha256=args.auxiliary_encoder_weights_sha256,
        )
        feature_losses = {}
        if args.moment_loss_enabled:
            feature_losses["moment"] = ConditionalMomentLoss(
                conditional_statistics["mean"],
                conditional_statistics["variance"],
                beta=args.moment_beta,
                min_group_size=args.moment_min_group_size,
                eps=args.moment_eps,
                kernel=args.moment_kernel,
                ema_enabled=args.moment_ema_enabled,
                ema_decay=args.moment_ema_decay,
            )
        if args.infonce_loss_enabled:
            moa_labels = load_condition_moa_labels(
                args.cellflux_metadata_path,
                conditional_statistics["condition_names"],
            )
            feature_losses["infonce"] = SupervisedPrototypeInfoNCELoss(
                conditional_statistics["mean"],
                moa_labels["prototype_moa_ids"],
                temperature=args.infonce_temperature,
                base_temperature=args.infonce_base_temperature,
            )
        auxiliary_manager = AuxiliaryLossManager(encoder, feature_losses).to(dist_util.dev())
        diffusion.configure_auxiliary_losses(
            auxiliary_manager,
            gate_thresh=args.moment_gate_thresh,
            gate_slope=args.moment_gate_slope,
        )
        if args.moment_loss_enabled:
            auxiliary_loss_schedules["moment"] = AuxiliaryLossSchedule(
                max_weight=args.lambda_moment_max,
                start_fraction=args.moment_ramp_start_fraction,
            )
        if args.infonce_loss_enabled:
            auxiliary_loss_schedules["infonce"] = AuxiliaryLossSchedule(
                max_weight=args.lambda_infonce_max,
                start_fraction=args.infonce_ramp_start_fraction,
                end_fraction=args.infonce_ramp_end_fraction,
            )
    if args.cfg_distill_loss_enabled:
        if args.dataset != "cellflux_bbbc021":
            raise ValueError("CFG distillation is currently supported only for cellflux_bbbc021")
        teacher_model = None
        if args.cfg_distill_teacher_mode == "frozen":
            teacher_checkpoint = Path(args.cfg_distill_teacher_checkpoint)
            if not teacher_checkpoint.is_file():
                raise FileNotFoundError(
                    f"frozen CFG teacher checkpoint does not exist: {teacher_checkpoint}"
                )
            if len(args.cfg_distill_teacher_checkpoint_sha256) != 64:
                raise ValueError("frozen CFG teacher requires an exact SHA-256")
            actual_teacher_sha256 = sha256_file(teacher_checkpoint)
            if actual_teacher_sha256 != args.cfg_distill_teacher_checkpoint_sha256:
                raise ValueError(
                    "frozen CFG teacher checkpoint SHA-256 "
                    f"{actual_teacher_sha256} != {args.cfg_distill_teacher_checkpoint_sha256}"
                )
            teacher_model = copy.deepcopy(model)
            teacher_model.load_state_dict(
                torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
            )
            teacher_model.to(dist_util.dev()).eval().requires_grad_(False)
            if dist.get_rank() == 0:
                logger.log(
                    "loaded frozen CFG teacher "
                    f"{teacher_checkpoint} sha256={actual_teacher_sha256}"
                )
        elif args.cfg_distill_teacher_mode == "online":
            if args.cfg_distill_teacher_checkpoint:
                raise ValueError("online CFG teacher must not name a frozen checkpoint")
            if args.cfg_distill_teacher_checkpoint_sha256:
                raise ValueError("online CFG teacher must not name a checkpoint SHA-256")
        else:
            raise ValueError(
                "cfg_distill_teacher_mode must be either 'online' or 'frozen'"
            )
        diffusion.configure_cfg_distillation(
            student_cfg_scale=args.cfg_distill_student_scale,
            teacher_cfg_scale=args.cfg_distill_teacher_scale,
            clip_denoised=args.cfg_distill_clip_denoised,
            teacher_model=teacher_model,
            pair_student_dropout=args.cfg_distill_pair_student_dropout,
        )
        auxiliary_loss_schedules["cfg_distill"] = AuxiliaryLossSchedule(
            max_weight=args.lambda_cfg_distill_max,
            start_fraction=args.cfg_distill_ramp_start_fraction,
            end_fraction=args.cfg_distill_ramp_end_fraction,
        )

    if dist.get_rank() == 0:
        wandb.watch(model, log="all")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}")
    else:
        batch_size = args.batch_size

    if dist.get_rank() == 0:
        logger.log("creating data loader...")

    data, test_data = load_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=batch_size,
        image_size=data_image_size,
        num_workers=args.num_workers,
        cellflux_metadata_path=args.cellflux_metadata_path,
        cellflux_embedding_path=args.cellflux_embedding_path,
        cellflux_compounds=args.cellflux_compounds,
        cellflux_excluded_compounds=args.cellflux_excluded_compounds,
        cellflux_max_eval_samples=args.cellflux_max_eval_samples,
        cellflux_condition_group_size=args.cellflux_condition_group_size,
        eval_batch_size=args.fid_batch_size,
    )
    if conditional_statistics is not None:
        if data.dataset.condition_names != list(conditional_statistics["condition_names"]):
            raise ValueError("training condition IDs do not match the conditional-statistics cache")
        dataset_counts = torch.tensor(
            [len(data.dataset.condition_to_indices[index]) for index in range(len(data.dataset.condition_names))],
            dtype=torch.long,
        )
        if not torch.equal(dataset_counts, conditional_statistics["counts"]):
            raise ValueError("training condition counts do not match the conditional-statistics cache")

    fid_eval_epochs = []
    if args.fid_eval_epochs:
        fid_eval_epochs = [int(item.strip()) for item in args.fid_eval_epochs.split(",") if item.strip()]
        if fid_eval_epochs != sorted(set(fid_eval_epochs)):
            raise ValueError("--fid_eval_epochs must be unique and sorted")
        if args.epochs <= 0 or any(epoch <= 0 or epoch > args.epochs for epoch in fid_eval_epochs):
            raise ValueError("--fid_eval_epochs entries must lie in [1, --epochs]")
        if args.dataset != "cellflux_bbbc021":
            raise ValueError("In-process matched FID is currently implemented only for cellflux_bbbc021")
    epoch_checkpoint_epochs = None
    if args.epoch_checkpoint_epochs:
        epoch_checkpoint_epochs = [
            int(item.strip()) for item in args.epoch_checkpoint_epochs.split(",") if item.strip()
        ]
        if epoch_checkpoint_epochs != sorted(set(epoch_checkpoint_epochs)):
            raise ValueError("--epoch_checkpoint_epochs must be unique and sorted")
        if args.epochs <= 0 or any(
            epoch <= 0 or epoch > args.epochs for epoch in epoch_checkpoint_epochs
        ):
            raise ValueError("--epoch_checkpoint_epochs entries must lie in [1, --epochs]")
        missing_fid_checkpoints = sorted(set(fid_eval_epochs) - set(epoch_checkpoint_epochs))
        if missing_fid_checkpoints:
            raise ValueError(
                "Every FID epoch must have a durable checkpoint; missing "
                f"{missing_fid_checkpoints} from --epoch_checkpoint_epochs"
            )
        if args.epochs not in epoch_checkpoint_epochs:
            raise ValueError("--epoch_checkpoint_epochs must include the final epoch")
    fid_evaluator = None
    if fid_eval_epochs:
        if args.fid_moa_eval:
            if not args.fid_retain_pngs:
                raise ValueError("--fid_moa_eval requires --fid_retain_pngs=True")
            for path in (args.fid_moa_repo, args.fid_moa_checkpoint, args.fid_moa_config):
                if not path or not Path(path).exists():
                    raise ValueError(f"Invalid inline MoA path: {path!r}")
        fid_evaluator = CellFluxFIDEvaluator(
            diffusion=diffusion,
            dataloader=test_data,
            output_dir=Path(workdir) / "fid",
            nfe=args.fid_nfe,
            sampler=args.fid_sampler,
            eta=args.fid_eta,
            order=args.fid_order,
            cfg_scale=args.fid_cfg_scale,
            seed=args.fid_seed,
            use_fp16=args.use_fp16,
            rho=args.fid_rho,
            retain_pngs=args.fid_retain_pngs,
            moa_eval_repo=args.fid_moa_repo if args.fid_moa_eval else None,
            moa_checkpoint=args.fid_moa_checkpoint if args.fid_moa_eval else None,
            moa_config=args.fid_moa_config if args.fid_moa_eval else None,
            moa_batch_size=args.fid_moa_batch_size,
            moa_num_workers=args.fid_moa_num_workers,
        )

    if args.use_augment:
        augment = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    else:
        augment = None

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        batch_size=batch_size,
        microbatch=-1 if args.microbatch >= batch_size else args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        augment_pipe=augment,
        train_mode=args.train_mode,
        resume_train_flag=resume_train_flag,
        class_drop_prob=args.class_drop_prob,
        epochs=args.epochs,
        epoch_checkpoint_epochs=epoch_checkpoint_epochs,
        fid_eval_epochs=fid_eval_epochs,
        fid_evaluator=fid_evaluator,
        max_steps=args.max_steps,
        auxiliary_loss_schedules=auxiliary_loss_schedules,
        auxiliary_group_size=(
            args.cellflux_condition_group_size if args.moment_loss_enabled else 0
        ),
        auxiliary_gradient_log_interval=args.auxiliary_gradient_log_interval,
        **sample_defaults(),
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset="edges2handbags",
        schedule_sampler="real-uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=256,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp="",
        workdir_root="workdir",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=8,
        use_augment=False,
        pretrained_ckpt=None,
        train_mode="ddbm",
        class_drop_prob=0.0,
        epochs=0,
        epoch_checkpoint_epochs="",
        fid_eval_epochs="",
        fid_batch_size=16,
        fid_nfe=20,
        fid_sampler="dbim",
        fid_eta=0.0,
        fid_order=2,
        fid_cfg_scale=0.0,
        fid_seed=42,
        fid_rho=7.0,
        fid_retain_pngs=False,
        fid_moa_eval=os.environ.get("CELLFLUX_FID_MOA_EVAL", "False") == "True",
        fid_moa_repo=os.environ.get("CELLFLUX_FID_MOA_REPO", ""),
        fid_moa_checkpoint=os.environ.get("CELLFLUX_FID_MOA_CHECKPOINT", ""),
        fid_moa_config=os.environ.get("CELLFLUX_FID_MOA_CONFIG", ""),
        fid_moa_batch_size=int(os.environ.get("CELLFLUX_FID_MOA_BATCH_SIZE", "32")),
        fid_moa_num_workers=int(os.environ.get("CELLFLUX_FID_MOA_NUM_WORKERS", "10")),
        max_steps=0,
        cellflux_metadata_path=None,
        cellflux_embedding_path=None,
        cellflux_compounds=None,
        cellflux_excluded_compounds=None,
        cellflux_max_eval_samples=0,
        cellflux_condition_group_size=0,
        moment_loss_enabled=False,
        auxiliary_encoder=FID_INCEPTION_ENCODER,
        auxiliary_encoder_weights_path="",
        auxiliary_encoder_weights_sha256="",
        moment_stats_path="",
        lambda_moment_max=0.1,
        moment_ramp_start_fraction=0.6,
        moment_beta=0.5,
        moment_min_group_size=4,
        moment_eps=1e-6,
        moment_kernel="diagonal_moments",
        moment_gate_thresh=0.0,
        moment_gate_slope=0.0,
        moment_ema_enabled=False,
        moment_ema_decay=0.9,
        infonce_loss_enabled=False,
        lambda_infonce_max=0.01,
        infonce_ramp_start_fraction=0.6,
        infonce_ramp_end_fraction=1.0,
        infonce_temperature=0.1,
        infonce_base_temperature=0.1,
        auxiliary_gradient_log_interval=0,
        cfg_distill_loss_enabled=False,
        lambda_cfg_distill_max=0.1,
        cfg_distill_ramp_start_fraction=0.6,
        cfg_distill_ramp_end_fraction=1.0,
        cfg_distill_student_scale=0.2,
        cfg_distill_teacher_scale=1.5,
        cfg_distill_clip_denoised=True,
        cfg_distill_teacher_mode="online",
        cfg_distill_teacher_checkpoint="",
        cfg_distill_teacher_checkpoint_sha256="",
        cfg_distill_pair_student_dropout=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
