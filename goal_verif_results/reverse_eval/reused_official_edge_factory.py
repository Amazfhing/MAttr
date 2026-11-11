def factory(model, graph, pretok, args, pick_index):
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    n_examples = len(pretok)
    real_flat = graph.real_edge_mask.flatten().bool()
    cumsum = (real_flat.float().cumsum(0).long() - 1).clamp(min=0).to(device)
    real_flat_device = real_flat.float().to(device)
    corrupt_topk = False
    corrupted_cache = {}
    cc_bytes = 0
    cc_max_bytes = int(args.cf_cache_gb * 1000000000.0)
    cc_stats = {'seen': 0, 'fwd': 0, 'full_logged': False}
    source_hooks = []
    input_node = graph.nodes['input']
    source_hooks.append(('hook_embed', 'input', graph.forward_index(input_node), False))
    for l in range(n_layers):
        head0 = graph.nodes[f'a{l}.h0']
        head0_idx = graph.forward_index(head0, attn_slice=False)
        source_hooks.append((f'blocks.{l}.attn.hook_result', f'attn_{l}', head0_idx, True))
        mlp_node = graph.nodes[f'm{l}']
        mlp_idx = graph.forward_index(mlp_node)
        source_hooks.append((f'blocks.{l}.hook_mlp_out', f'm{l}', mlp_idx, False))

    def loss_fn(mask_flat):
        nonlocal cc_bytes
        idx = pick_index()
        clean_tokens, attention_mask, n_pos, corrupted_tokens, labels = pretok[idx]
        correct_idx, incorrect_idx = (labels[0], labels[1])
        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            return None
        cc_stats['seen'] += 1
        corrupted_acts = corrupted_cache.get(idx)
        if corrupted_acts is None:
            cc_stats['fwd'] += 1
            corrupted_acts = {}

            def make_corrupted_hook(name, fwd_idx, is_attn):

                def hook(act, hook):
                    if is_attn:
                        for h in range(n_heads):
                            corrupted_acts[fwd_idx + h] = act[:, :, h].detach()
                    else:
                        corrupted_acts[fwd_idx] = act.detach()
                return hook
            corrupted_fwd_hooks = [(hname, make_corrupted_hook(nname, fidx, is_a)) for hname, nname, fidx, is_a in source_hooks]
            with torch.no_grad():
                model.run_with_hooks(corrupted_tokens, fwd_hooks=corrupted_fwd_hooks, attention_mask=attention_mask)
            nbytes = sum((v.numel() * v.element_size() for v in corrupted_acts.values()))
            if cc_bytes + nbytes <= cc_max_bytes:
                corrupted_cache[idx] = corrupted_acts
                cc_bytes += nbytes
            elif not cc_stats['full_logged']:
                cc_stats['full_logged'] = True
                logger.info('corrupted-acts cache full at %.2f GB (%d examples); further examples recompute each step', cc_bytes / 1000000000.0, len(corrupted_cache))
        expanded = mask_flat[cumsum]
        full_mask = (expanded * real_flat_device).view(graph.n_forward, graph.n_backward)
        corruption_mask = (full_mask if corrupt_topk else 1 - full_mask).to(model.cfg.dtype)
        clean_acts = {}

        def make_clean_hook(name, fwd_idx, is_attn):

            def hook(act, hook):
                if is_attn:
                    for h in range(n_heads):
                        clean_acts[fwd_idx + h] = act[:, :, h]
                else:
                    clean_acts[fwd_idx] = act
                return act
            return hook
        diff_cache = {}
        zeros_cache = {}

        def diff_for(src_i, n_pos, dtype):
            cached = diff_cache.get(src_i)
            if cached is not None:
                return cached
            if src_i not in corrupted_acts:
                z = zeros_cache.get((n_pos, dtype))
                if z is None:
                    z = torch.zeros(1, n_pos, d_model, device=device, dtype=dtype)
                    zeros_cache[n_pos, dtype] = z
                return z
            clean = clean_acts.get(src_i)
            if clean is None:
                return corrupted_acts[src_i] - torch.zeros_like(corrupted_acts[src_i])
            d = corrupted_acts[src_i] - clean
            diff_cache[src_i] = d
            return d

        def _stack_and_weight(weights, *diffs):
            stack = torch.stack(diffs, dim=2)
            if weights.dim() == 1:
                return einsum(stack, weights, 'batch pos src hidden, src -> batch pos hidden')
            return einsum(stack, weights, 'batch pos src hidden, src heads -> batch pos heads hidden')

        def make_dest_hook(dest_node, letter=None):
            prev_idx = graph.prev_index(dest_node)
            bwd_idx = graph.backward_index(dest_node, qkv=letter, attn_slice=True)
            weights = corruption_mask[:prev_idx, bwd_idx]

            def hook(activations, hook):
                diffs = [diff_for(src_i, activations.shape[1], activations.dtype) for src_i in range(prev_idx)]
                update = checkpoint(_stack_and_weight, weights, *diffs, use_reentrant=False)
                return activations + update
            return hook
        clean_fwd_hooks = [(hname, make_clean_hook(nname, fidx, is_a)) for hname, nname, fidx, is_a in source_hooks]
        dest_hooks = []
        for l in range(n_layers):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{l}.h0']
                dest_hooks.append((node.qkv_inputs[i], make_dest_hook(node, letter=letter)))
            node = graph.nodes[f'm{l}']
            dest_hooks.append((node.in_hook, make_dest_hook(node)))
        node = graph.nodes['logits']
        dest_hooks.append((node.in_hook, make_dest_hook(node)))
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_fwd_hooks + dest_hooks, attention_mask=attention_mask)
        logit_diff = logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]
        return logit_diff.float() if corrupt_topk else -logit_diff.float()
    return loss_fn