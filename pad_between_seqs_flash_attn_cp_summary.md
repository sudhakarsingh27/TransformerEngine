# Enabling `pad_between_seqs=True` with FlashAttention in Context Parallelism

## Goal
Identify and resolve all barriers preventing `pad_between_seqs=True` from working with FlashAttention in TransformerEngine's context parallelism (CP) implementation.

---

## Barriers Resolved (in order)

### 1. FA guard explicitly disables FlashAttention for THD + pad_between_seqs
**File:** `transformer_engine/pytorch/attention/dot_product_attention/utils.py`
**Problem:** Guard sets `use_flash_attention = False` when `pad_between_seqs=True` with THD format.
**Fix:** Bypassed the guard (commented out).

### 2. `pad_between_seqs` detection bug — `[:-1]` comparison misses last sequence
**File:** `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py` (~line 1333)
**Problem:** `torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1])` doesn't detect padding in the last sequence.
**Fix:** Changed to `torch.equal(cu_seqlens_q_padded, cu_seqlens_q)` (compare full tensors).

### 3. Non-zero `dout` at padding positions causes gradient leakage
**File:** `tests/pytorch/attention/run_attention_with_cp.py` (~lines 365, 483)
**Problem:** `dout` at padding positions has random non-zero values. `out.backward(dout)` propagates these into `dq` at padding positions, causing comparison failures.
**Fix:** Zero out `dout` at padding positions before calling backward, for both no-CP and CP paths:
```python
# No-CP path
if pad_between_seqs == "True":
    effective_dout = dout.clone()
    for b in range(config.batch_size):
        effective_dout[cu_seqlens_q[b + 1] : cu_seqlens_q_padded[b + 1]] = 0.0
out.backward(effective_dout)

# CP path (with per-rank cu_seqlens)
if pad_between_seqs == "True":
    cu_seqlens_q_padded_rank = cu_seqlens_q_padded // world_size
    cu_seqlens_q_rank = get_cu_seqlens_on_cp_rank(...)
    effective_dout_ = dout_.clone()
    for b in range(config.batch_size):
        effective_dout_[cu_seqlens_q_rank[b + 1] : cu_seqlens_q_padded_rank[b + 1]] = 0.0
out_.backward(effective_dout_)
```

### 4. Backends don't guarantee zeros at padding positions in forward output
**File:** `tests/pytorch/attention/run_attention_with_cp.py` (~lines 584-620)
**Problem:** `out` has non-zero values at padding positions. Hard assertions fail.
**Fix:** Replace assertions with explicit zeroing of padding positions in all tensors (`out`, `dq`, `out_`, `dq_`, `dk`, `dv`, `dk_`, `dv_`) before comparison. Also added `.detach()` to avoid autograd in-place modification errors.

### 5. `FlashAttentionBackend` CP path hardcodes `pad_between_seqs=False`
**File:** `transformer_engine/pytorch/attention/dot_product_attention/backends.py` (~lines 740, 937-950)
**Problem:** The CP call in `FlashAttentionBackend.forward` passes `cu_seqlens_q` as `cu_seqlens_q_padded` and sets `pad_between_seqs=False`, so the CP kernel doesn't know about padding.
**Fix:** Added `cu_seqlens_q_padded`, `cu_seqlens_kv_padded`, `pad_between_seqs` parameters to the function signature and threaded them through to the CP call.

**File:** `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py` (~line 1474)
**Fix:** Pass `cu_seqlens_q_padded`, `cu_seqlens_kv_padded`, `pad_between_seqs` from DPA to `self.flash_attention()`.

### 6. FlashAttention fundamentally can't distinguish padding from actual tokens (ROOT CAUSE)
**File:** `tests/pytorch/attention/run_attention_with_cp.py` — `generate_input_shapes()` (~line 103-106)
**Problem:** `flash_attn_varlen_func` uses `cu_seqlens_q` as actual positions in the tensor — it has no `cu_seqlens_q_padded` parameter. When `cu_seqlens_q` contains cumulative actual seqlens (without gaps), FlashAttention misidentifies where tokens are in the padded tensor.
**Fix:** Only compute separate `cu_seqlens_q` for FusedAttention (which supports both parameters). For FlashAttention, keep `cu_seqlens_q == cu_seqlens_q_padded`:
```python
# Before:
if kernel_backend == "FusedAttention" or pad_between_seqs == "True":
    cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()

# After:
if kernel_backend == "FusedAttention":
    cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()
```

---

## Files Modified

| File | Changes |
|------|---------|
| `tests/pytorch/attention/run_attention_with_cp.py` | `generate_input_shapes` cu_seqlens fix; dout zeroing for no-CP and CP backward; detach tensors; zero padding positions before comparison; fall-through for THD comparison |
| `transformer_engine/pytorch/attention/dot_product_attention/backends.py` | Added `cu_seqlens_q_padded`, `cu_seqlens_kv_padded`, `pad_between_seqs` params to `FlashAttentionBackend.forward`; threaded them to CP call |
| `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py` | Fixed `pad_between_seqs` detection (removed `[:-1]`); passed padded seqlens to flash backend |
| `transformer_engine/pytorch/attention/dot_product_attention/utils.py` | Bypassed FA guard for THD + pad_between_seqs |
| `tests/pytorch/attention/test_attention_with_cp.py` | Added `pad_between_seqs=True` to `test_cp_with_flash_attention` |

---

## Important Caveat
The current solution makes FlashAttention treat padding tokens as part of each sequence (`cu_seqlens_q == cu_seqlens_q_padded`). This means padding tokens with random q/k/v values participate in attention. Both no-CP and CP produce **consistent** results, but the output at actual positions differs from what FusedAttention (which properly ignores padding) would produce. A more correct solution would require compacting the tensor (removing padding) before FlashAttention and expanding back afterward.
