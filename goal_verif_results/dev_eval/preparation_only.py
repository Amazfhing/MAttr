def prepare(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    model_dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Select one authorised GPU by UUID.'
    device = torch.device('cuda:0')
    workdir = Path(args.workdir).resolve()
    repo = Path(args.repo).resolve() if args.repo else workdir / 'code/matryoshka-attribution'
    root = workdir / 'results'
    root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='feedback_mattr_eval_', dir=str(root)))
    logger.info('OUTPUT DIRECTORY: %s', out)
    dump_json(out / 'arguments.json', vars(args))
    (out / 'repo_commit.txt').write_text(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True))
    factory = official_factory(repo / 'scripts/mib/eval_mib_edge.py', out)
    baseline_dir = Path(args.baseline_dir).resolve()
    old_summary = json.loads((baseline_dir / 'summary.json').read_text())
    assert old_summary['model'] == MODEL and old_summary['resolved_revision'] == REVISION
    snapshot = Path(old_summary['snapshot'])
    assert snapshot.is_dir(), 'Pinned local model snapshot is missing.'
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    labels = {s: tokenizer.encode(s, add_special_tokens=False) for s in ['A', 'B']}
    assert all((len(x) == 1 for x in labels.values()))
    labels = {s: x[0] for s, x in labels.items()}
    assert labels == old_summary['label_ids']
    baseline_version = old_summary.get('version', 'v1')
    data_version = args.data_version if args.data_version else baseline_version
    baseline_data_name = old_summary.get('data_name', 'feedback_retry_answer')
    data_name = args.data_name if args.data_name else baseline_data_name
    if data_version:
        data_root = workdir / f'data/{data_name}_{data_version}'
    else:
        data_root = workdir / f'data/{data_name}_v1'
    train = read_pairs(data_root / 'train.jsonl', tokenizer, labels, device)
    development = read_pairs(data_root / 'dev.jsonl', tokenizer, labels, device)
    saved_tokens = {r['pair_id']: r for r in map(json.loads, (baseline_dir / 'tokenized_pairs.jsonl').read_text().splitlines())}
    if not args.skip_count_check:
        expected_counts = {'v1': (64, 32), 'v2': (256, 64)}
        if data_name == 'feedback_retry_answer' and data_version in expected_counts:
            expected_train, expected_dev = expected_counts[data_version]
            assert len(train) == expected_train and len(development) == expected_dev, f'Expected {expected_train} train / {expected_dev} dev pairs for {data_name} {data_version}, got {len(train)} / {len(development)}'
        elif data_name == 'export_goal_verification' and data_version == 'v1':
            assert len(train) == 256 and len(development) == 64, f'Expected 256 train / 64 dev pairs for export_goal_verification v1, got {len(train)} / {len(development)}'
        else:
            logger.warning('No count validation for %s %s; loaded %d train / %d dev pairs', data_name, data_version, len(train), len(development))
    for example in development:
        old = saved_tokens[example['pair']['pair_id']]
        for side in ['base', 'source']:
            assert example['tokens'][side][0].tolist() == old[side]['input_ids'], 'Prompt/token mismatch with baseline.'
    evaluate_examples = development[:args.eval_limit]
    longest_train = max(range(len(train)), key=lambda i: train[i]['pretok'][2])
    parity_examples = development + [train[longest_train]]
    logger.info('Comparison precision: %s; full FP32 matmul precision requested.', args.dtype)
    logger.info('Tokenised %d train / %d dev pairs; evaluating first %d dev pairs, no filtering.', len(train), len(development), len(evaluate_examples))
    logger.info('Loading pinned HF model for conversion and numerical comparison.')
    hf = AutoModelForCausalLM.from_pretrained(str(snapshot), dtype=model_dtype, attn_implementation='eager', local_files_only=True).to(device).eval()
    hf.requires_grad_(False)
    hf_margins = [(e, side, raw_margin(hf, e, side, hf=True)) for e in parity_examples for side in ['base', 'source']]
    from matryoshka_attribution.deps import add_mib_to_sys_path
    add_mib_to_sys_path()
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    logger.info('Moving HF model to CPU before conversion to avoid two GPU-resident copies.')
    hf.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    logger.info('Converting the SAME checkpoint to HookedTransformer; no weight folding/centering.')
    model = HookedTransformer.from_pretrained(MODEL, hf_model=hf, tokenizer=tokenizer, revision=REVISION, device='cuda', dtype=model_dtype, fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False, refactor_factored_attn_matrices=False)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    model.requires_grad_(False)
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    parity = []
    for example, side, reference in hf_margins:
        actual = raw_margin(model, example, side)
        parity.append({'pair_id': example['pair']['pair_id'], 'side': side, 'HF_margin': reference, 'TL_margin': actual, 'absolute_error': abs(reference - actual), 'decision_disagrees': reference * actual < 0, 'either_margin_is_zero': reference == 0 or actual == 0})
    csv_write(out / 'hf_tl_comparison.csv', parity)
    max_error = max((r['absolute_error'] for r in parity))
    disagreements = sum((r['decision_disagrees'] for r in parity))
    logger.info('HF/TL comparison: max margin difference %.6f, %d/%d strict non-tie preference reversals.', max_error, disagreements, len(parity))
    if max_error > args.max_margin_error or disagreements > args.max_decision_disagreements:
        raise RuntimeError('HF/TL comparison failed the explicit pilot tolerances. Scores have NOT been trained.')
    dev_ids = {e['pair']['pair_id'] for e in development}
    dev_comparisons = [r for r in parity if r['pair_id'] in dev_ids]

    def accuracy_report(field):
        decisions = {}
        by_action = {}
        lookup = {e['pair']['pair_id']: e['pair'] for e in development}
        for r in dev_comparisons:
            correct = r[field] > 0 if r['side'] == 'base' else r[field] < 0
            decisions.setdefault(r['pair_id'], []).append(correct)
            action = lookup[r['pair_id']][r['side'] + '_action']
            by_action.setdefault(action, []).append(correct)
        return {'contexts': len(dev_comparisons), 'pairwise_action_accuracy': sum((sum(v) for v in decisions.values())) / len(dev_comparisons), 'both_sides_correct_fraction': sum((all(v) for v in decisions.values())) / len(decisions), 'per_action_accuracy': {k: sum(v) / len(v) for k, v in by_action.items()}}
    precision_baseline = {'dtype': args.dtype, 'model': MODEL, 'revision': REVISION, 'HF': accuracy_report('HF_margin'), 'TransformerLens': accuracy_report('TL_margin'), 'max_margin_error': max_error, 'strict_preference_reversals': disagreements, 'comparison_contexts': len(parity), 'note': 'New execution baseline; do not substitute the earlier BF16 accuracy.'}
    dump_json(out / 'precision_baseline.json', precision_baseline)
    print('\nPRECISION COMPARISON SUMMARY\n' + json.dumps(precision_baseline, indent=2), flush=True)
    if args.conversion_only:
        logger.info('CONVERSION CHECK PASSED. No interventions or training were run. Results: %s', out)
        return
    graph = Graph.from_model(model)
    total = int(graph.real_edge_mask.sum())
    logger.info('Graph: %d scored edges; all model parameters frozen.', total)
    return (model, graph, evaluate_examples, factory, out, precision_baseline)