"""Packed/page metadata adaptations to the original sparse attention API."""
from pathlib import Path
from .runtime import candidate, validate
from .native import _dense_route, native_ops


def packed_attention(case):
    validate(case)
    import torch
    op, p = case._call.signature.name, case.params
    dense, sources = _dense_route(p)

    def run(q, k, v, *args):
        if op == "GQAVarlenFwdOp":
            cq, ck, *optional = args
            qb, kb = cq.tolist(), ck.tolist()
            lengths = [b-a for a, b in zip(kb, kb[1:])]
            table = None
        elif op == "GQAPagedFwdOp":
            table, lengths, cq, *optional = args
            qb, lengths = cq.tolist(), lengths.tolist()
        else:
            lengths, table = args
            lengths, optional = lengths.tolist(), []
            qb = [i*q.shape[1] for i in range(q.shape[0]+1)]
            q = q.reshape(-1, *q.shape[2:])
            k, v = (x.reshape(-1, p["page_size"], *x.shape[1:]) for x in (k, v))
        optional = [*optional, *([None]*6)][:6]
        _, _, _, cos, sin, _ = optional
        results = []
        for b, length in enumerate(lengths):
            qi = q[qb[b]:qb[b+1]].unsqueeze(0)
            if not qi.shape[1]:
                continue
            if table is None:
                ki, vi = k[kb[b]:kb[b+1]], v[kb[b]:kb[b+1]]
            else:
                pages = table[b, :(length+k.shape[1]-1)//k.shape[1]].long()
                ki, vi = (x.index_select(0, pages).reshape(-1, *x.shape[2:])[:length] for x in (k, v))
            results.append(dense(qi, ki.unsqueeze(0), vi.unsqueeze(0), None, None, None,
                                 None if cos is None else cos[:length],
                                 None if sin is None else sin[:length]).squeeze(0))
        out = torch.cat(results, 0)
        return out.reshape(p["o"][0])

    return candidate(run, "npu_sparse_flash_attention (packed/paged GQA)",
        "Per-request sequence slices or page-table gathers feed the complete original shared-KV attention; right-aligned causal/window ranges and supplied RoPE tables stay inside run. No scores/softmax are calculated by Torch.",
        [Path(__file__), *sources], timing_mode="host_completed")


def mla_decode(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    ops, sources = native_ops("csrc/attention/sparse_flash_attention")

    def run(q, q_pe, kv, k_pe):
        b, h, dim = q.shape
        sk, kh, tail = kv.shape[1], kv.shape[2], q_pe.shape[-1]
        group = h//kh
        pg = 128 if group > 64 else group
        def query(x, width):
            x = x.reshape(b, kh, group, x.shape[-1])
            return F.pad(x, (0, width-x.shape[-1], 0, pg-group)).reshape(b*kh, 1, pg, width)
        def key(x, width):
            return F.pad(x.permute(0, 2, 1, 3).contiguous(), (0, width-x.shape[-1])).reshape(b*kh, sk, 1, width)
        qi, qr, ki, kr = query(q, 512), query(q_pe, 64), key(kv, 512), key(k_pe, 64)
        indices = torch.arange(sk, device=q.device, dtype=torch.int32).reshape(1, 1, 1, sk).expand(b*kh, 1, 1, sk).contiguous()
        out, _, _ = ops.npu_sparse_flash_attention(qi, ki, ki, indices, (dim+tail)**-0.5,
            query_rope=qr, key_rope=kr, sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
            sparse_mode=0, attention_mode=2, return_softmax_lse=False)
        return out[:, 0, :group, :dim].reshape(b, h, dim)

    return candidate(run, "npu_sparse_flash_attention (MLA decode)",
        "Original API's shared KV and separate RoPE coordinates match MLA decode. Fold KV heads into batch, pad features/heads with zeros, retain original scale and crop only padding.", [Path(__file__), *sources])


def dsa_decode(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    ops, sources = native_ops("csrc/attention/sparse_flash_attention")
    p = case.params

    def run(q, kv, indices):
        b, sq, h, total = q.shape
        sk, kh, tail = kv.shape[1], kv.shape[2], p["dim_tail"]
        dim, group, stride = total-tail, h//kh, p["stride_kv"]
        pg = 128 if group > 64 else group
        start = p.get("q_start_index_s")
        start = sk*stride-sq if start is None else start
        metadata = indices.tolist()
        result = torch.full((b, sq, h, dim), float("nan"), device=q.device, dtype=q.dtype)
        for bi in range(b):
            for gi in range(kh):
                rows = []
                for ti in range(sq):
                    selected = {i for i in metadata[bi][ti][gi]
                                if 0 <= i < sk and stride-1+i*stride <= start+ti}
                    # Preserve the original manifest's forced first key and its
                    # boolean-scatter deduplication, including invalid indices.
                    if ti < stride-1:
                        selected.add(0)
                    if selected:
                        rows.append((ti, sorted(selected)))
                if not rows:
                    continue
                keys = kv[bi, :, gi]
                ki = F.pad(keys[:, :dim], (0, 512-dim)).reshape(1, sk, 1, 512)
                kr = F.pad(keys[:, dim:], (0, 64-tail)).reshape(1, sk, 1, 64)
                max_width = max(len(selected) for _, selected in rows)
                row_limit = max(1, min(256, (64*1024*1024)//(4*max_width)))
                for begin in range(0, len(rows), row_limit):
                    part = rows[begin:begin+row_limit]
                    tids = torch.tensor([ti for ti, _ in part], device=q.device, dtype=torch.long)
                    query = q[bi].index_select(0, tids)[:, gi*group:(gi+1)*group]
                    qi = F.pad(query[..., :dim], (0, 512-dim, 0, pg-group)).unsqueeze(0)
                    qr = F.pad(query[..., dim:], (0, 64-tail, 0, pg-group)).unsqueeze(0)
                    width = max(len(selected) for _, selected in part)
                    # Original return_softmax_lse=True + sparse_mode=0 counts
                    # each nonnegative prefix before the -1 padded index tail.
                    selected_rows = [selected + [-1]*(width-len(selected)) for _, selected in part]
                    local = torch.tensor(selected_rows, device=q.device, dtype=torch.int32).reshape(1, len(part), 1, width)
                    out, _, _ = ops.npu_sparse_flash_attention(qi, ki, ki, local,
                        total**-0.5 if p.get("sm_scale") is None else p["sm_scale"], query_rope=qr,
                        key_rope=kr, sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                        sparse_mode=0, attention_mode=2, return_softmax_lse=True)
                    result[bi, tids, gi*group:(gi+1)*group] = out[0, :, :group, :dim]
        return result

    return candidate(run, "npu_sparse_flash_attention (DSA complete selected rows)",
        "Read and deduplicate original selection metadata, retain compressed causal bounds and forced key zero. Batch independent complete query domains with sorted valid index prefixes and -1 tails through the original return_softmax_lse API branch; no attention arithmetic or partial-domain reduction outside the owned API. Preserve fully masked NaNs, pad/crop features and heads only.",
        [Path(__file__), *sources], timing_mode="host_completed")


def merge_states(case):
    validate(case)
    import torch
    import triton
    from .runtime import TRITON_ROOT, kernel_module
    native, hw, sources = kernel_module(TRITON_ROOT + "dcp/dcp_a2a.py", ("_fused_dcp_lse_combine_kernel",))

    def run(v_a, s_a, v_b, s_b):
        # Both the explicit manifest formula and original DCP API use ln.
        recv = torch.stack((torch.cat((v_a.float(), s_a.unsqueeze(-1)), -1),
                            torch.cat((v_b.float(), s_b.unsqueeze(-1)), -1)), 0).contiguous()
        tokens, heads, dim = v_a.shape
        output = torch.empty((tokens, heads, dim+1), dtype=torch.float32, device=v_a.device)
        rows = tokens*heads
        native._fused_dcp_lse_combine_kernel[(min(rows, hw.get_vectorcore_num()),)](
            recv, output, recv, recv, 0, 0, 0, 0, 0, *recv.stride(), *output.stride(),
            dim, heads, rows, DCP_SIZE=2, SCATTER_TOKENS=True, LSE_PACK_DIM=1,
            BLOCK_D=triton.next_power_of_2(dim), RETURN_LSE=True, HAS_LOCAL=False)
        return output[..., :dim].to(v_a.dtype).contiguous(), output[..., dim].contiguous()

    return candidate(run, "_fused_dcp_lse_combine_kernel(RETURN_LSE=True)",
        "Pack the two FP32 partial states and natural-log LSE in the original receive-buffer layout. Invoke the intact JIT entry with original host grid/BLOCK_D settings and return both merged value and LSE, including zero output/-inf LSE for two empty states. Distributed module registration is omitted; kernel body is unchanged.",
        [Path(__file__), *sources])


def _lse_row_route():
    import torch
    import torch.nn.functional as F
    from .runtime import VENDOR
    ops, sources = native_ops("csrc/attention/sparse_flash_mla")
    sources += [x for x in (VENDOR / "csrc/attention/sparse_flash_mla_metadata").rglob("*") if x.is_file() and x.suffix in {".cpp", ".h"}]

    def row(q, k, v, scale):
        # q[heads,K], k/v[length,KVheads,K/V]. Fold KV heads into batch.
        h, kd = q.shape
        length, kh, vd = k.shape[0], k.shape[1], v.shape[-1]
        group = h//kh
        qi = F.pad(q.reshape(kh, 1, group, kd), (0, 512-kd)).contiguous()
        kv = F.pad(torch.cat((k, v), -1).permute(1, 0, 2).unsqueeze(2), (0, 512-kd-vd)).contiguous()
        used_q = torch.ones(kh, dtype=torch.int32, device=q.device)
        used_k = torch.full((kh,), length, dtype=torch.int32, device=q.device)
        metadata = ops.npu_sparse_flash_mla_metadata(group, 1, 512,
            seqused_q=used_q, seqused_ori_kv=used_k, batch_size=kh,
            max_seqlen_q=1, max_seqlen_ori_kv=length, ori_mask_mode=0,
            layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=False)
        out, lse = ops.npu_sparse_flash_mla(qi, ori_kv=kv, metadata=metadata,
            seqused_q=used_q, seqused_ori_kv=used_k, softmax_scale=scale,
            ori_mask_mode=0, layout_q="BSND", layout_kv="BSND", return_softmax_lse=True)
        return out[..., kd:kd+vd].reshape(h, vd), lse.reshape(h)

    return row, sources


def lse_attention(case):
    validate(case)
    import torch
    row, sources = _lse_row_route()
    p, op = case.params, case._call.signature.name

    def run(q, k, *args):
        if op == "MLAVarlenFwdOp":
            pe, v, offsets = args
            # Broadcast shared positional key coordinates as a layout operation.
            k = torch.cat((k, pe.unsqueeze(1).expand(-1, k.shape[1], -1)), -1)
            offsets = offsets.tolist()
            chunk_offsets = offsets
        else:
            v, offsets, chunk_offsets, _token_indices = args
            offsets = offsets.tolist()
            # Read actual metadata each call; the manifest guarantees consistency.
            chunk_offsets = chunk_offsets.tolist()
        output = torch.zeros(p["o"][0], dtype=q.dtype, device=q.device)
        lse = torch.zeros(p["lse"][0], dtype=getattr(torch, p["lse"][1]), device=q.device)
        scale = p.get("scale") if op == "NSACompressedVarlenFwdOp" else p.get("sm_scale")
        scale = q.shape[-1]**-0.5 if scale is None else scale
        for b, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            start = chunk_offsets[b]
            for token in range(begin, end):
                if op == "NSACompressedVarlenFwdOp":
                    length = (token-begin+1)//p["bs"]
                else:
                    length = token-begin+1 if p["is_causal"] else end-begin
                if length:
                    out, logsum = row(q[token], k[start:start+length], v[start:start+length], scale)
                    output[token], lse[token] = out, logsum.to(lse.dtype)
        return output, lse

    return candidate(run, "npu_sparse_flash_mla + owned metadata API (complete output/LSE)",
        "Pack [K,V,0] and [Q,0] into original 512-wide shared-KV API, folding KV heads into batch. Per-query full causal prefix or fully closed compressed blocks; original kernel returns both values and natural-log LSE. Generate original native metadata within each run. BF16 only; no FP16 narrowing.",
        [Path(__file__), *sources], timing_mode="host_completed")


def nsa_selected(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    p = case.params
    ops, sources = native_ops("csrc/attention/sparse_flash_attention")

    def run(q, k, v, blocks, counts, offsets, token_indices):
        metadata, count, bounds = blocks.tolist(), counts.tolist(), offsets.tolist()
        dim, vd = k.shape[-1], v.shape[-1]
        output = torch.zeros((q.shape[0], q.shape[1], vd), device=q.device, dtype=q.dtype)
        group, bs = q.shape[1]//k.shape[1], p["block_size"]
        pg = 128 if group > 64 else group
        scale = dim**-0.5 if p["scale"] is None else p["scale"]
        for begin, end in zip(bounds[:-1], bounds[1:]):
            for head in range(k.shape[1]):
                rows = []
                for t in range(begin, end):
                    limit = t-begin if p["is_causal"] else end-begin-1
                    # NSA's original reference gathers slots directly. Preserve
                    # both their order and multiplicity, including repeated blocks.
                    selected = [i for block in metadata[t][head][:max(0, count[t][head])]
                                for i in range(block*bs, (block+1)*bs) if 0 <= i <= limit]
                    if selected:
                        rows.append((t, selected))
                if not rows:
                    continue
                max_width = max(len(ids) for _, ids in rows)
                # Original ComputeS2LoopInfo clips the index capacity to the
                # physical KV row count. Padding unused rows preserves repeated
                # slots when their complete multiplicity exceeds sequence length.
                storage_rows = max(end-begin, max_width)
                kv = F.pad(torch.cat((k[begin:end, head], v[begin:end, head]), -1),
                           (0, 512-dim-vd, 0, storage_rows-(end-begin))).reshape(1, storage_rows, 1, 512)
                kr = torch.zeros((*kv.shape[:-1], 64), device=q.device, dtype=q.dtype)
                row_limit = max(1, min(256, (64*1024*1024)//(4*max_width)))
                for first in range(0, len(rows), row_limit):
                    part = rows[first:first+row_limit]
                    tids = torch.tensor([t for t, _ in part], device=q.device, dtype=torch.long)
                    query = q.index_select(0, tids)[:, head*group:(head+1)*group]
                    qi = F.pad(query, (0, 512-dim, 0, pg-group)).unsqueeze(0)
                    qr = torch.zeros((*qi.shape[:-1], 64), device=q.device, dtype=q.dtype)
                    width = max(len(ids) for _, ids in part)
                    # Validity is a nonnegative prefix; value ordering need not
                    # change. The original API binary-searches only the -1 tail.
                    selected_rows = [ids+[-1]*(width-len(ids)) for _, ids in part]
                    indices = torch.tensor(selected_rows, device=q.device, dtype=torch.int32).reshape(1, len(part), 1, width)
                    out, _, _ = ops.npu_sparse_flash_attention(qi, kv, kv, indices, scale,
                        query_rope=qr, key_rope=kr, sparse_block_size=1,
                        layout_query="BSND", layout_kv="BSND", sparse_mode=0,
                        attention_mode=2, return_softmax_lse=True)
                    output[tids, head*group:(head+1)*group] = out[0, :, :group, dim:dim+vd]
        return output

    return candidate(run, "npu_sparse_flash_attention (NSA complete selected slots)",
        "Expand original selected blocks/counts into complete local token slots, preserving order, multiplicity, sequence and causal boundaries. Batch independent queries with valid index prefixes and -1 tails through the unchanged owned sparse attention API. Pack [K,V,0] and [Q,0], crop only value coordinates, and retain zero outputs for empty selections; no attention-score math outside the target.",
        [Path(__file__), *sources], timing_mode="host_completed")


def paged_append(case):
    validate(case)
    import torch
    from workloads.rope import rope_frequency_tables
    from .runtime import TRITON_ROOT, WORKSPACE, VENDOR, module
    from .cache import copy_payload
    p = case.params
    dense, sources = _dense_route(p)
    copier, extra = module(TRITON_ROOT + "kda_state_copy_kernel.py")
    sources += [*extra, Path(__file__).with_name("cache.py"), VENDOR / "vllm_ascend/ops/kda_state_copy_plan.py"]
    if p["fuse_rope"]:
        rotation, extra = module(TRITON_ROOT + "rope.py", hardware=True)
        sources += [*extra, WORKSPACE / "TileOPs/workloads/rope.py"]

    def run(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, offsets, lengths, table):
        bounds, lengths, pages = offsets.tolist(), lengths.tolist(), table.tolist()
        size = p["page_size"]
        output, append_k, append_v, append_slots = [], [], [], []
        if p["fuse_rope"]:
            dim = p.get("rotary_dim") or q.shape[-1]
            cos, sin = rope_frequency_tables(dim, p["max_position"], base=p["rope_base"], dtype=q.dtype, device=q.device)
            tables = torch.cat((cos, sin), -1)
        for b, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            old = lengths[b]
            old_slots = torch.tensor([pages[b][i//size]*size+i%size for i in range(old)], device=q.device, dtype=torch.long)
            old_k, old_v = (x.index_select(0, old_slots) for x in (k_pages, v_pages))
            qi, ki = q[begin:end], k_new[begin:end]
            if p["fuse_rope"]:
                pos = torch.arange(old, old+end-begin, dtype=torch.int64, device=q.device)
                qi, ki = (rotation.rope_forward_triton_siso(x.clone().contiguous(), cos_sin_cache=tables,
                    positions=pos, rope_dim=dim, is_neox_style=True) for x in (qi, ki))
            all_k, all_v = torch.cat((old_k, ki)), torch.cat((old_v, v_new[begin:end]))
            output.append(dense(qi.unsqueeze(0), all_k.unsqueeze(0), all_v.unsqueeze(0)).squeeze(0))
            append_k.append(ki)
            append_v.append(v_new[begin:end])
            append_slots.extend(pages[b][i//size]*size+i%size for i in range(old, old+end-begin))
        slots = torch.tensor(append_slots, device=q.device, dtype=torch.int64)
        kc = copy_payload(copier, k_pages.reshape(-1, size, *k_pages.shape[1:]), torch.cat(append_k), slots, True)
        vc = copy_payload(copier, v_pages.reshape(-1, size, *v_pages.shape[1:]), torch.cat(append_v), slots, True)
        return torch.cat(output), kc.reshape(k_pages.shape), vc.reshape(v_pages.shape)

    return candidate(run, "npu_sparse_flash_attention + _kda_state_copy_kernel (paged append)",
        "Gather original cache prefixes; generate manifest RoPE table state and rotate only appended Q/K with the original kernel. Attend original full prefixes plus new rows, then native-scatter appends into private full cache copies. Return attention and both entire caches. FP8 writes and logit softcap excluded.",
        [Path(__file__), *sources], timing_mode="host_completed")
