"""NanoGPT speedrun track 1 (record #92, ANVIL2) on ONE GPU: the RTX 5090 port (rtx-local-v2).

Launch:  bash run_rtx.sh                    (full run: warmup, then the timed run and final validation)
         RTX_PREFLIGHT=1 bash run_rtx.sh    (memory preflight: warmup + one validation forward, then exit)

The model, optimizer, schedules and kernels are the record's (track_1_short/). What one GPU changes lives
in track_1_short/portable/ (see its __init__ and docs/rtx-local-v2.md): each step runs the record's 8
ranks as K sequential microbatches, attention is FlexAttention instead of the Hopper-only FA3 build, the
n-gram table is whole on the GPU at reduced size, and nothing is captured as a CUDA graph.
"""
import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("PORTABLE", "1")
assert os.environ["PORTABLE"] == "1", "train_gpt_rtx.py is the PORTABLE=1 trainer"
# Plain `python train_gpt_rtx.py` works too: a one-process group needs only these.
for _key, _value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"),
                     ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29517")):
    os.environ.setdefault(_key, _value)

from track_1_short.run_log import log_environment, read_source, start_run_log

# Read the source ASAP, for the run log.
code = read_source(sys.argv[0])

import copy
import gc
import time

import torch

torch.empty(1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
from torch import nn

from track_1_short.canonical_mask import BackgroundCanonicalMask
from track_1_short.config import (
    LR_COOLDOWN_FRAC,
    SPLIT_EMBED_STAGE,
    TRAINING_STAGES,
    WS_POST_YARN_EXT,
    Hyperparameters,
)
from track_1_short.data import cu_seqlens_rows
from track_1_short.distributed import setup_distributed
from track_1_short.model.gpt import GPT
from track_1_short.model.prefix_prediction import build_prefix_table_bucket
from track_1_short.ngram_table import NGRAM_DIM, NGRAM_VOCAB_SIZE, is_update_step
from track_1_short.optim.anvil import anvil_bank_update
from track_1_short.perf.kernels.mlp import mlp_block_n, prime_stage_cache
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.portable import config as rtx
from track_1_short.portable.attention import build_window_masks
from track_1_short.portable.data import MicrobatchLoader, microbatch_cu_seqlens_rows, validation_batches
from track_1_short.portable.grads import GradAccumulator
from track_1_short.portable.memory import MemoryReport, gib
from track_1_short.portable.sampled import PortableSampledSoftmax
from track_1_short.portable.tables import LocalNgramTable, LocalTables, LocalValueEmbeds
from track_1_short.schedule import TrainingSchedule
from track_1_short.tail_average import TailAverages
from track_1_short.training import TrainingManager

dynamo.config.recompile_limit = 64
PRINT_EVERY = 25
PREFLIGHT = os.environ.get("RTX_PREFLIGHT") == "1"


def to_cpu(obj):
    """Deep copy with every tensor on the host: the warmup backup must not double GPU memory."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


class Trainer:
    """Everything one training step touches, built once in main()."""

    def __init__(self, model, uncompiled, training_manager, sampled, ngram, value_embeds, grads, device):
        self.model = model
        self.uncompiled = uncompiled
        self.training_manager = training_manager
        self.sampled = sampled
        self.ngram = ngram
        self.value_embeds = value_embeds
        self.grads = grads
        self.device = device

    def train_step(self, step: int, batches, tables: LocalTables) -> torch.Tensor:
        """One optimizer step over the step's microbatches; returns the summed training loss (device).
        `tables` decides whether this step is an update event of the n-gram table and value_embeds."""
        tm, model = self.training_manager, self.model
        do_adam = tm.is_adam_step(step)
        windows = tm.get_forward_args()
        loss_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        for batch in batches:
            tokens = batch.inputs.numel()
            masks = build_window_masks(batch.cum_seqlens, tokens, windows.ws_short, windows.ws_long)
            # After the last fp8 refresh (end of the previous step), so the gathered rows are current.
            sampled_loss = self.sampled.prepare(step, batch.targets_cpu, self.uncompiled.lm_head_f8_col)
            sink = self.ngram.grad_sink(tokens)
            loss = model(batch.inputs, batch.targets, batch.cum_seqlens, batch.ngram_ids, tm.get_forward_args(sampled_loss),
                         ngram_sink=sink, value_embed_grad=self.value_embeds.grad_accum, window_masks=masks).sum()
            loss.backward()
            self.ngram.accumulate_grad(batch.ngram_ids, sink.grad)
            sink.grad = None
            self.grads.collect()
            loss_sum += loss.detach()
        self.grads.release(do_adam)
        deferred = tm.step_optimizers(step, tables, frozenset())
        assert not deferred
        # The record refreshes the fp8 weight caches right before the next forward (deferred_gathers.py);
        # with nothing in flight here, right after the update is the same point. lm_head only changes on
        # Adam steps, so only those refresh its fp8 copies.
        self.uncompiled.quantize_attn_fp8()
        self.uncompiled.quantize_mlp_fp8(refresh_lm=do_adam)
        return loss_sum


def main():
    args = Hyperparameters()
    env = setup_distributed()
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
    device = env.device
    mb_tokens = rtx.MICROBATCH_TOKENS
    for stage in TRAINING_STAGES:
        assert stage.batch_size % mb_tokens == 0, f"batch {stage.batch_size} is not a multiple of {mb_tokens}-token microbatches"
    val_seq = rtx.VAL_SEQ_LEN
    assert args.val_tokens % val_seq == 0

    staging = PinnedBatchStaging(max(mb_tokens, val_seq),
                                 max(microbatch_cu_seqlens_rows(mb_tokens), cu_seqlens_rows(val_seq)), device)
    schedule = TrainingSchedule(
        TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, device=device,
        cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT,
    )
    total_steps = schedule.total_steps if rtx.MAX_STEPS is None else min(rtx.MAX_STEPS, schedule.total_steps)
    full_run = total_steps == schedule.total_steps

    print0, flush_log = start_run_log(True, args.run_id)
    log_environment(print0, code)
    memory = MemoryReport(print0, device)
    microbatches = [s.batch_size // mb_tokens for s in TRAINING_STAGES]
    print0(f"rtx-local-v2: one GPU ({torch.cuda.get_device_name(device)}), {mb_tokens}-token microbatches "
           f"{microbatches} per step by stage, n-gram table {NGRAM_VOCAB_SIZE:,} rows (1/{rtx.NGRAM_DIVISOR} of the "
           f"record's), validation {val_seq}-token forwards, MLP tile 128x{mlp_block_n()}, eval CE slab "
           f"{rtx.EVAL_CE_SLAB_ROWS} rows, steps {total_steps}/{schedule.total_steps}"
           f"{' (PREFLIGHT)' if PREFLIGHT else ''}", console=True)
    flush_log()

    ########################################
    #          Model and optimizer         #
    ########################################
    model: nn.Module = GPT(
        vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768, max_seq_len=val_seq,
        ngram_dim=NGRAM_DIM, world_size=1, device=device,
    ).cuda()
    model.cast_matrix_weights_bf16()
    ngram = LocalNgramTable(max_events=schedule.total_steps, device=device)
    # The forward reads the table at the row ids themselves (portable/tables.py): it IS the row cache.
    model.ngram_cache = ngram.table
    memory.mark("after_model_construction")
    model.quantize_attn_fp8()
    model.quantize_mlp_fp8(refresh_lm=True)
    prime_stage_cache()
    sampled = PortableSampledSoftmax(schedule, model.vocab_size, model.lm_head.in_features, mb_tokens, device)

    uncompiled = model
    model = torch.compile(model, dynamic=False, fullgraph=True)
    training_manager = TrainingManager(model, schedule, bank_update=anvil_bank_update)
    optimizer = training_manager.optimizer
    value_embeds = LocalValueEmbeds(uncompiled.value_embeds, optimizer, device)
    grads = GradAccumulator(optimizer.param_cfgs)
    tail_averages = TailAverages(optimizer, 0, value_embed_updates=is_update_step, total_steps=schedule.total_steps)
    trainer = Trainer(model, uncompiled, training_manager, sampled, ngram, value_embeds, grads, device)
    memory.mark("after_optimizer_construction")
    memory.component("n-gram table + gradient + Adam state", ngram.nbytes)
    memory.component("fp32 gradient accumulators", grads.nbytes)
    memory.component("value_embeds gradient buffer", value_embeds.grad_accum.nbytes)

    ########################################
    #            Warmup kernels            #
    ########################################
    print0("Compiling model and warming up kernels (several minutes on first execution)", console=True)
    initial_state = dict(model=to_cpu(uncompiled.state_dict()), optimizer=to_cpu(optimizer.state_dict()))
    # Each stage start and each sampled-softmax count change is a new compiled graph: visit both sides
    # of every transition, and the first Adam steps after the embed split (a new optimizer path).
    transition_steps = sorted(set(training_manager.get_transition_steps()) | set(sampled.count_change_steps()))
    split_step = schedule.split_step
    warmup_steps = sorted({0, 1} | {s + o for s in transition_steps for o in (-2, -1, 0, 1) if s + o >= 2}
                          | {split_step + o for o in (1, 2, 3) if split_step + o < schedule.total_steps})
    missing = sampled.stage_keys(range(schedule.total_steps)) - sampled.stage_keys(warmup_steps)
    assert not missing, f"warmup never compiles (P, stage) {sorted(missing)}"
    print0(f"Warmup steps {warmup_steps}, {rtx.WARMUP_MICROBATCHES} microbatches each", console=True)
    # During warmup the tables update on every Adam step (as the record's warmup does), so their update
    # kernels and value_embeds' compiled Adam (both beta pairs) are built before the clock.
    warmup_tables = LocalTables(ngram, value_embeds, training_manager.is_adam_step)
    warmup_loader = MicrobatchLoader(args.train_files, mb_tokens, staging)
    warmup_start = time.perf_counter()
    for position, step in enumerate(warmup_steps):
        training_manager.advance_schedule(step)
        stage, _ = schedule.lookup(step)
        batches = warmup_loader.next_step(rtx.WARMUP_MICROBATCHES, stage.train_max_seq_len)
        trainer.train_step(step, batches, warmup_tables)
        if position == 0:
            memory.mark("after_first_warmup_step")
    torch.cuda.synchronize()
    print0(f"Warmup done in {time.perf_counter() - warmup_start:.0f}s", console=True)
    memory.mark("after_warmup")
    del warmup_loader

    if PREFLIGHT:
        # One validation forward at full length: the eval graph's peak, which training never reaches.
        model.eval()
        uncompiled.complete_yarn_tables()
        batch = next(validation_batches(args.val_files, args.val_tokens, val_seq, staging))
        cfg = training_manager.get_forward_args()
        with torch.no_grad():
            masks = build_window_masks(batch.cum_seqlens, val_seq, cfg.ws_short, cfg.ws_long)
            loss = model(batch.inputs, batch.targets, batch.cum_seqlens, batch.ngram_ids, cfg, window_masks=masks).mean()
        print0(f"PREFLIGHT validation forward: {val_seq} tokens, loss {loss.item():.4f} (after warmup updates)", console=True)
        memory.mark("after_validation_forward")
        ok = memory.verdict()
        print0(f"PREFLIGHT {'PASS' if ok else 'FAIL'}", console=True)
        flush_log()
        dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    print0("Resetting model", console=True)
    model.zero_grad(set_to_none=True)
    grads.reset()
    uncompiled.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    ngram.reset()
    value_embeds.reset()
    sampled.reset()
    del initial_state
    uncompiled.rearm_fp8_bootstrap()
    uncompiled.quantize_attn_fp8()
    uncompiled.quantize_mlp_fp8(refresh_lm=True)
    model.train()

    ########################################
    #        Training and validation       #
    ########################################
    loader = MicrobatchLoader(args.train_files, mb_tokens, staging)
    tables = LocalTables(ngram, value_embeds, is_update_step)
    canon_mask_builder = BackgroundCanonicalMask(model.vocab_size, owner=True, print0=print0)
    gc.collect()
    gc.freeze()
    gc.disable()
    uncompiled.limit_yarn_rebuild(mb_tokens)

    training_time_ms = 0
    loss_window = torch.zeros((), dtype=torch.float32, device=device)
    tokens_window = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    canon_mask_builder.start()
    # Prefix-token table: one process builds every bucket (the record splits them across ranks).
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size, bucket=0, num_buckets=1))
    sampled.set_prefix_table(model.prefix_table.cpu().numpy())
    for step in range(total_steps + 1):
        last_step = step == total_steps
        training_manager.advance_schedule(step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step:
                training_manager.apply_final_ws_ext()
                canon_mask_builder.wait()
                canon_mask_builder.collect(model.canon_mask)
                if full_run:
                    tail_averages.ship()  # on the clock, as in the record
            uncompiled.complete_yarn_tables()
            # Reading the validation data is on the clock (as in the record); the forwards are not.
            val_batches = list(validation_batches(args.val_files, args.val_tokens, val_seq, staging))
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            cfg = training_manager.get_forward_args()
            val_loss = torch.zeros((), dtype=torch.float32, device=device)
            with torch.no_grad():
                for batch in val_batches:
                    masks = build_window_masks(batch.cum_seqlens, val_seq, cfg.ws_short, cfg.ws_long)
                    val_loss += model(batch.inputs, batch.targets, batch.cum_seqlens, batch.ngram_ids, cfg,
                                      window_masks=masks).mean()
            val_loss /= len(val_batches)
            del val_batches
            print0(f"step:{step}/{total_steps} val_loss:{val_loss.item():.6f} train_time:{training_time_ms:.0f}ms "
                   f"step_avg:{training_time_ms / max(step, 1):.2f}ms", console=True)
            flush_log()
            gc.collect()
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            break

        stage, _ = schedule.lookup(step)
        batches = loader.next_step(stage.batch_size // mb_tokens, stage.train_max_seq_len)
        loss_window += trainer.train_step(step, batches, tables)
        tokens_window += stage.batch_size
        if full_run:
            tail_averages.tick(step)

        if (step + 1) % PRINT_EVERY == 0 or step + 1 >= total_steps - 1:
            approx_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"step:{step + 1}/{total_steps} train_loss_per_token:{loss_window.item() / tokens_window:.4f} "
                   f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / (step + 1):.2f}ms", console=True)
            loss_window.zero_()
            tokens_window = 0

    gc.enable()
    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    memory.verdict()
    print0(f"final_val_loss={val_loss.item():.6f} training_time_ms={training_time_ms:.0f} "
           f"peak_memory_allocated={gib(torch.cuda.max_memory_allocated())}", console=True)
    flush_log()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
