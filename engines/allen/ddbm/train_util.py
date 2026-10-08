import copy
import functools
import os
import random
import time
from contextlib import contextmanager

import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW, RAdam

from . import dist_util, logger
from .nn import update_ema

from ddbm.random_util import get_generator

import glob

from .tracking import wandb


def create_training_optimizer(parameters, name, lr, betas, weight_decay):
    """Build the optimizer named by a dataset-aligned formal recipe."""
    normalized_name = str(name).lower()
    if normalized_name not in {"adamw", "radam"}:
        raise ValueError(
            f"optimizer_name must be 'adamw' or 'radam', got {name!r}"
        )
    normalized_betas = tuple(float(beta) for beta in betas)
    if len(normalized_betas) != 2 or not all(
        0.0 <= beta < 1.0 for beta in normalized_betas
    ):
        raise ValueError(f"invalid optimizer betas: {normalized_betas}")
    optimizer_class = AdamW if normalized_name == "adamw" else RAdam
    return optimizer_class(
        parameters,
        lr=float(lr),
        betas=normalized_betas,
        weight_decay=float(weight_decay),
    )


def add_cellflux_condition(model_kwargs, extra, class_drop_prob=0.0, training=True):
    """Add the molecular condition, with CellFlux's batchwise CFG dropout."""
    if not 0.0 <= class_drop_prob <= 1.0:
        raise ValueError(f"class_drop_prob must be in [0, 1], got {class_drop_prob}")
    if not isinstance(extra, dict) or "condition" not in extra:
        return False
    dropped = training and class_drop_prob > 0.0 and torch.rand(()) < class_drop_prob
    if not dropped:
        model_kwargs["condition"] = extra["condition"]
    return bool(dropped)


def get_rank_cuda_rng_state():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state(device=dist_util.dev())


def set_rank_cuda_rng_state(cuda_rng):
    if not torch.cuda.is_available() or cuda_rng is None:
        return
    if isinstance(cuda_rng, (list, tuple)):
        cuda_rng = cuda_rng[int(os.environ["LOCAL_RANK"])]
    torch.cuda.set_rng_state(cuda_rng, device=dist_util.dev())


def _flatten_parameter_gradients(gradients, parameters):
    """Flatten unscaled parameter gradients, materializing unused entries as zero."""
    chunks = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            chunks.append(
                torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32)
            )
        else:
            chunks.append(gradient.detach().reshape(-1).to(dtype=torch.float32))
    return torch.cat(chunks)


def _gradient_interaction_statistics(main_gradient, auxiliary_gradient):
    """Return exact main/aux statistics for the DDP-averaged batch gradients."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(main_gradient, op=dist.ReduceOp.SUM)
        dist.all_reduce(auxiliary_gradient, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        main_gradient.div_(world_size)
        auxiliary_gradient.div_(world_size)

    main_norm = torch.linalg.vector_norm(main_gradient)
    auxiliary_norm = torch.linalg.vector_norm(auxiliary_gradient)
    dot = torch.dot(main_gradient, auxiliary_gradient)
    if main_norm.item() > 0.0 and auxiliary_norm.item() > 0.0:
        cosine = dot / (main_norm * auxiliary_norm)
    else:
        cosine = dot.new_zeros(())
    ratio = auxiliary_norm / main_norm if main_norm.item() > 0.0 else main_norm.new_zeros(())
    return {
        "main_grad_norm": main_norm.item(),
        "aux_grad_norm": auxiliary_norm.item(),
        "main_aux_grad_dot": dot.item(),
        "main_aux_grad_cosine": cosine.item(),
        "main_aux_grad_conflict": float(dot.item() < 0.0),
        "aux_to_main_grad_norm_ratio": ratio.item(),
    }


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        optimizer_name="radam",
        optimizer_beta1=0.9,
        optimizer_beta2=0.999,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        train_mode="ddbm",
        resume_train_flag=False,
        class_drop_prob=0.0,
        epochs=0,
        epoch_checkpoint_epochs=None,
        fid_eval_epochs=(),
        fid_evaluator=None,
        max_steps=0,
        auxiliary_loss_schedules=None,
        auxiliary_loss_budget_schedule=None,
        auxiliary_group_size=0,
        auxiliary_gradient_log_interval=0,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.optimizer_name = str(optimizer_name).lower()
        self.optimizer_betas = (float(optimizer_beta1), float(optimizer_beta2))
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        self.epochs = int(epochs)
        self.epoch_checkpoint_epochs = (
            None
            if epoch_checkpoint_epochs is None
            else set(int(epoch) for epoch in epoch_checkpoint_epochs)
        )
        self.fid_eval_epochs = set(int(epoch) for epoch in fid_eval_epochs)
        self.fid_evaluator = fid_evaluator
        self.max_steps = int(max_steps)
        self.auxiliary_loss_schedules = dict(auxiliary_loss_schedules or {})
        self.auxiliary_loss_budget_schedule = auxiliary_loss_budget_schedule
        self.auxiliary_group_size = int(auxiliary_group_size)
        self.auxiliary_gradient_log_interval = int(auxiliary_gradient_log_interval)
        if self.auxiliary_gradient_log_interval < 0:
            raise ValueError("auxiliary_gradient_log_interval must be non-negative")
        if self.auxiliary_gradient_log_interval and not self.auxiliary_loss_schedules:
            raise ValueError("auxiliary gradient logging requires an auxiliary loss")
        self.auxiliary_total_steps = self.epochs * len(self.data) if self.epochs > 0 else self.max_steps
        if self.auxiliary_loss_schedules and self.auxiliary_total_steps <= 0:
            raise ValueError("auxiliary loss schedules require a finite training horizon")
        if self.auxiliary_loss_budget_schedule is not None:
            if not self.auxiliary_loss_schedules:
                raise ValueError("auxiliary budget schedule requires an auxiliary loss")
            if self.auxiliary_total_steps <= 0:
                raise ValueError("auxiliary budget schedule requires a finite training horizon")
        if self.auxiliary_group_size:
            if self.auxiliary_group_size < 2:
                raise ValueError("condition-grouped auxiliary losses require a group size of at least two")
            if self.batch_size % self.auxiliary_group_size != 0:
                raise ValueError("rank-local batch must contain whole condition groups")
            if self.microbatch % self.auxiliary_group_size != 0:
                raise ValueError("microbatch must contain whole condition groups")
        self.completed_epoch = 0
        self.attempted_steps = 0

        self.train_mode = train_mode
        if not 0.0 <= class_drop_prob <= 1.0:
            raise ValueError(f"class_drop_prob must be in [0, 1], got {class_drop_prob}")
        self.class_drop_prob = class_drop_prob

        self.step = 0
        self.resume_train_flag = resume_train_flag
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        self._load_and_sync_parameters()
        if not self.resume_train_flag:
            self.resume_step = 0

        self.opt = create_training_optimizer(
            self.model.parameters(),
            self.optimizer_name,
            self.lr,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay,
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
            self._load_training_state()
        else:
            self.ema_params = [copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))]

        if torch.cuda.is_available():
            self.use_ddp = True
            local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                # The CellFlux U-Net has no buffers. Explicitly disable forward-time
                # collectives so rank-local CFG dropout may retain its original
                # semantics even when an active CFG auxiliary branch adds a paired
                # unconditional forward only on conditional ranks.
                broadcast_buffers=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs["generator"], self.batch_size, 42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            if self.resume_train_flag:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log("Resume step: ", self.resume_step)

            self.model.load_state_dict(torch.load(resume_checkpoint, map_location="cpu"))
            self.model.to(dist_util.dev())

            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = torch.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]

            dist.barrier()
        elif self.resume_train_flag:
            raise FileNotFoundError(
                f"Missing EMA state for strict resume at step {self.resume_step}, rate {rate}"
            )
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split("/")[-1].startswith("freq"):
            prefix = "freq_"
        else:
            prefix = ""
        opt_checkpoint = os.path.join(os.path.dirname(main_checkpoint), f"{prefix}opt_{self.resume_step:06}.pt")
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()
        elif self.resume_train_flag:
            raise FileNotFoundError(f"Missing optimizer state for strict resume: {opt_checkpoint}")

    def _training_state_path(self, step=None, for_preemption=False):
        step = self.resume_step if step is None else step
        prefix = "freq_" if for_preemption else ""
        return os.path.join(
            get_blob_logdir(),
            f"{prefix}train_state_rank{dist.get_rank():03d}_{step:06d}.pt",
        )

    def _load_training_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        for_preemption = os.path.basename(main_checkpoint).startswith("freq_")
        path = self._training_state_path(step=self.resume_step, for_preemption=for_preemption)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing rank-local training state for strict resume: {path}")
        state = torch.load(path, map_location="cpu")
        if state["step"] != self.resume_step:
            raise ValueError(f"Training-state step {state['step']} != checkpoint step {self.resume_step}")
        self.completed_epoch = int(state["completed_epoch"])
        self.attempted_steps = int(state["attempted_steps"])
        self.scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        # Older checkpoints stored all visible devices. The helper selects
        # only this process's device instead of overwriting every GPU RNG.
        set_rank_cuda_rng_state(state["cuda_rng"])
        auxiliary_state = state.get("auxiliary_loss_state", {})
        if hasattr(self.diffusion, "load_auxiliary_runtime_state_dict"):
            self.diffusion.load_auxiliary_runtime_state_dict(auxiliary_state)
        elif auxiliary_state:
            raise ValueError("checkpoint has auxiliary state but diffusion cannot restore it")
        batch_sampler = getattr(self.data, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(state["sampler_epoch"])
        dist.barrier()

    def run_loop(self):
        if self.epochs > 0:
            return self.run_epoch_loop()
        while True:
            for batch, cond, extra in self.data:

                if "inpaint" in self.workdir:
                    _, mask, label = extra
                else:
                    mask = None

                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                if self.augment is not None:
                    batch, _augment_labels = self.augment(batch)
                if isinstance(cond, torch.Tensor) and batch.ndim == cond.ndim:
                    cond = {"xT": cond}
                else:
                    cond["xT"] = cond["xT"]
                if mask is not None:
                    # cond["mask"] = mask
                    cond["y"] = label
                condition_dropped = add_cellflux_condition(
                    cond,
                    extra,
                    class_drop_prob=self.class_drop_prob,
                    training=True,
                )
                logger.logkv_mean("condition_dropped", float(condition_dropped))
                if isinstance(extra, dict):
                    for auxiliary_key in ("condition_id", "plate_id", "structure_id"):
                        if auxiliary_key in extra:
                            cond[f"_aux_{auxiliary_key}"] = extra[auxiliary_key]

                took_step = self.run_step(batch, cond, auxiliary_enabled=not condition_dropped)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                    test_batch, test_cond, test_extra = next(iter(self.test_data))
                    if "inpaint" in self.workdir:
                        _, mask, label = test_extra
                    else:
                        mask = None
                    if isinstance(test_cond, torch.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_cond = {"xT": test_cond}
                    else:
                        test_cond["xT"] = test_cond["xT"]
                    if mask is not None:
                        # test_cond["mask"] = mask
                        test_cond["y"] = label
                    add_cellflux_condition(
                        test_cond,
                        test_extra,
                        class_drop_prob=self.class_drop_prob,
                        training=False,
                    )
                    self.run_test_step(test_batch, test_cond)
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.max_steps and self.step >= self.max_steps:
                    return

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)

    def run_epoch_loop(self):
        if self.completed_epoch > self.epochs:
            raise ValueError(f"Checkpoint epoch {self.completed_epoch} exceeds requested epochs {self.epochs}")
        self._evaluate_fid_if_due(self.completed_epoch)
        for epoch_index in range(self.completed_epoch, self.epochs):
            completed_epoch = epoch_index + 1
            if dist.get_rank() == 0:
                logger.log(f"starting epoch {completed_epoch}/{self.epochs}")
            batches_seen = 0
            for batch, cond, extra in self.data:
                batches_seen += 1
                if self.augment is not None:
                    batch, _augment_labels = self.augment(batch)
                if isinstance(cond, torch.Tensor) and batch.ndim == cond.ndim:
                    cond = {"xT": cond}
                else:
                    cond["xT"] = cond["xT"]
                condition_dropped = add_cellflux_condition(
                    cond,
                    extra,
                    class_drop_prob=self.class_drop_prob,
                    training=True,
                )
                logger.logkv_mean("condition_dropped", float(condition_dropped))
                if isinstance(extra, dict):
                    for auxiliary_key in ("condition_id", "plate_id", "structure_id"):
                        if auxiliary_key in extra:
                            cond[f"_aux_{auxiliary_key}"] = extra[auxiliary_key]
                took_step = self.run_step(batch, cond, auxiliary_enabled=not condition_dropped)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()
                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

            if batches_seen != len(self.data):
                raise RuntimeError(f"Epoch consumed {batches_seen} batches, expected {len(self.data)}")
            self.completed_epoch = completed_epoch
            if self.epoch_checkpoint_epochs is None or completed_epoch in self.epoch_checkpoint_epochs:
                self.save(completed_epoch=completed_epoch)
            self._evaluate_fid_if_due(completed_epoch)
            logs = logger.dumpkvs()
            if dist.get_rank() == 0 and logs:
                wandb.log(logs, step=self.step)

    def _evaluate_fid_if_due(self, completed_epoch):
        if completed_epoch not in self.fid_eval_epochs:
            return
        if self.fid_evaluator is None:
            raise RuntimeError(f"FID scheduled at e{completed_epoch} but no evaluator is configured")
        if hasattr(self.fid_evaluator, "is_complete") and self.fid_evaluator.is_complete(completed_epoch):
            if dist.get_rank() == 0:
                logger.log(f"matched FID for e{completed_epoch} is already complete; skipping")
        else:
            with self.ema_scope(0):
                self.fid_evaluator(self.model, completed_epoch)
        if hasattr(self.fid_evaluator, "run_jump_retrieval_if_due"):
            self.fid_evaluator.run_jump_retrieval_if_due(completed_epoch)
        if hasattr(self.fid_evaluator, "run_allencell_retrieval_if_due"):
            self.fid_evaluator.run_allencell_retrieval_if_due(completed_epoch)

    @contextmanager
    def ema_scope(self, ema_index=0):
        backup = [parameter.detach().clone() for parameter in self.model.parameters()]
        try:
            with torch.no_grad():
                for parameter, ema_parameter in zip(self.model.parameters(), self.ema_params[ema_index]):
                    parameter.copy_(ema_parameter)
            yield
        finally:
            with torch.no_grad():
                for parameter, original in zip(self.model.parameters(), backup):
                    parameter.copy_(original)

    def _current_auxiliary_loss_weights(self, enabled=True):
        weights = {
            name: schedule.weight(self.step, self.auxiliary_total_steps) if enabled else 0.0
            for name, schedule in self.auxiliary_loss_schedules.items()
        }
        for name, value in weights.items():
            logger.logkv_mean(f"lambda_{name}", value)
        return weights

    def _current_auxiliary_loss_budget_ratio(self, enabled=True):
        if not enabled:
            ratio = 0.0
        elif getattr(self, "auxiliary_loss_budget_schedule", None) is None:
            ratio = None
        else:
            ratio = self.auxiliary_loss_budget_schedule.ratio(
                self.step,
                self.auxiliary_total_steps,
            )
        if ratio is not None:
            logger.logkv_mean("aux_budget_ratio", ratio)
        return ratio

    def run_step(self, batch, cond, auxiliary_enabled=True):
        self.attempted_steps += 1
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        step_started = time.perf_counter()
        self.forward_backward(batch, cond, auxiliary_enabled=auxiliary_enabled)
        logger.logkv_mean("lg_loss_scale", np.log2(self.scaler.get_scale()))
        self.scaler.unscale_(self.opt)

        def _compute_norms():
            grad_norm = 0.0
            param_norm = 0.0
            for p in self.model.parameters():
                with torch.no_grad():
                    param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                    if p.grad is not None:
                        grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
            return np.sqrt(grad_norm), np.sqrt(param_norm)

        if self.attempted_steps % self.log_interval == 0:
            grad_norm, param_norm = _compute_norms()
            logger.logkv_mean("grad_norm", grad_norm)
            logger.logkv_mean("param_norm", param_norm)

        scale_before_step = self.scaler.get_scale()
        self.scaler.step(self.opt)
        self.scaler.update()
        took_step = self.scaler.get_scale() >= scale_before_step
        logger.logkv_mean("amp_step_skipped", float(not took_step))
        if took_step:
            self.step += 1
            self._update_ema()
            self._anneal_lr()
        step_seconds = time.perf_counter() - step_started
        logger.logkv_mean("step_seconds", step_seconds)
        logger.logkv_mean("samples_per_second", self.global_batch / step_seconds)
        if torch.cuda.is_available():
            logger.logkv_mean("cuda_peak_allocated_gb", torch.cuda.max_memory_allocated() / 2**30)
        self.log_step()
        return took_step

    def run_test_step(self, batch, cond):
        with torch.no_grad():
            self.forward_backward(batch, cond, train=False, auxiliary_enabled=False)

    def forward_backward(self, batch, cond, train=True, auxiliary_enabled=True):
        if train:
            self.opt.zero_grad()
        assert batch.shape[0] % self.microbatch == 0
        num_microbatches = batch.shape[0] // self.microbatch
        diagnose_auxiliary_gradients = (
            train
            and self.auxiliary_gradient_log_interval > 0
            and self.attempted_steps % self.auxiliary_gradient_log_interval == 0
        )
        diagnostic_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        main_gradient = None
        auxiliary_gradient = None
        for i in range(0, batch.shape[0], self.microbatch):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
                auxiliary_context = {
                    key[len("_aux_") :]: value
                    for key, value in micro_cond.items()
                    if key.startswith("_aux_")
                }
                micro_cond = {
                    key: value for key, value in micro_cond.items() if not key.startswith("_aux_")
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                auxiliary_loss_weights = self._current_auxiliary_loss_weights(
                    enabled=train and auxiliary_enabled
                )
                auxiliary_loss_budget_ratio = self._current_auxiliary_loss_budget_ratio(
                    enabled=train and auxiliary_enabled
                )

                if self.train_mode == "ddbm":
                    loss_kwargs = {"model_kwargs": micro_cond}
                    if self.auxiliary_loss_schedules:
                        loss_kwargs.update(
                            auxiliary_context=auxiliary_context,
                            auxiliary_loss_weights=auxiliary_loss_weights,
                        )
                        if auxiliary_loss_budget_ratio is not None:
                            loss_kwargs["auxiliary_loss_budget_ratio"] = (
                                auxiliary_loss_budget_ratio
                            )
                    compute_losses = functools.partial(
                        self.diffusion.training_bridge_losses,
                        self.ddp_model,
                        micro,
                        t,
                        **loss_kwargs,
                    )
                else:
                    raise NotImplementedError()

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                loss = (losses["loss"] * weights).mean() / num_microbatches
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {self.train_mode} loss at attempted step {self.attempted_steps}")

                if diagnose_auxiliary_gradients:
                    main_objective = (losses["mse"] * weights).mean() / num_microbatches
                    main_gradients = torch.autograd.grad(
                        main_objective,
                        diagnostic_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    microbatch_main_gradient = _flatten_parameter_gradients(
                        main_gradients, diagnostic_parameters
                    )
                    del main_gradients
                    if "aux_total" in losses:
                        auxiliary_objective = (
                            losses["aux_total"] * weights
                        ).mean() / num_microbatches
                        auxiliary_gradients = torch.autograd.grad(
                            auxiliary_objective,
                            diagnostic_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        microbatch_auxiliary_gradient = _flatten_parameter_gradients(
                            auxiliary_gradients, diagnostic_parameters
                        )
                        del auxiliary_gradients
                    else:
                        # A batchwise CFG-dropout step has no auxiliary graph.
                        # Keep every rank in the same all-reduce and represent
                        # that rank's auxiliary contribution exactly as zero.
                        microbatch_auxiliary_gradient = torch.zeros_like(
                            microbatch_main_gradient
                        )
                    if main_gradient is None:
                        main_gradient = microbatch_main_gradient
                        auxiliary_gradient = microbatch_auxiliary_gradient
                    else:
                        main_gradient.add_(microbatch_main_gradient)
                        auxiliary_gradient.add_(microbatch_auxiliary_gradient)
            log_loss_dict(self.diffusion, t, {k if train else "test_" + k: v * weights for k, v in losses.items()})
            if train:
                self.scaler.scale(loss).backward()

        if main_gradient is not None:
            statistics = _gradient_interaction_statistics(main_gradient, auxiliary_gradient)
            for key, value in statistics.items():
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite auxiliary gradient statistic {key} at attempted step "
                        f"{self.attempted_steps}"
                    )
                logger.logkv_mean(key, value)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("attempted_steps", self.attempted_steps)
        logger.logkv("samples", self.step * self.global_batch)

    def save(self, for_preemption=False, completed_epoch=None):
        def maybe_delete_earliest(filename):
            wc = filename.split(f"{(self.step):06d}")[0] + "*"
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3000:
                earliest = min(freq_states, key=lambda x: x.split("_")[-1].split(".")[0])
                os.remove(earliest)

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params):
            state_dict = self.model.state_dict()
            for i, (name, _) in enumerate(self.model.named_parameters()):
                assert name in state_dict
                state_dict[name] = params[i]
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)

                with open(os.path.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)

            with open(os.path.join(get_blob_logdir(), filename), "wb") as f:
                torch.save(self.opt.state_dict(), f)

        batch_sampler = getattr(self.data, "batch_sampler", None)
        sampler_epoch = int(getattr(batch_sampler, "epoch", self.completed_epoch))
        training_state = {
            "step": self.step,
            "completed_epoch": self.completed_epoch if completed_epoch is None else int(completed_epoch),
            "attempted_steps": self.attempted_steps,
            "scaler": self.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": get_rank_cuda_rng_state(),
            "sampler_epoch": sampler_epoch,
            "auxiliary_loss_state": (
                self.diffusion.auxiliary_runtime_state_dict()
                if hasattr(self.diffusion, "auxiliary_runtime_state_dict")
                else {}
            ),
        }
        torch.save(training_state, self._training_state_path(step=self.step, for_preemption=for_preemption))
        dist.barrier()

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/EMA/rank-local RNG state isn't saved.
        save_checkpoint(0, list(self.model.parameters()))
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split("/")[-1].startswith("freq"):
        prefix = "freq_"
    else:
        prefix = ""
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
