"""Export this project's saved GPT-2 IOI edge scores without loading model weights."""
import argparse
import csv
import inspect
import json
from pathlib import Path
import tempfile

import torch
from matryoshka_attribution.deps import add_mib_to_sys_path


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run_dir).resolve()
    saved = torch.load(run / "circuit/ioi_gpt2_scores.pt",
                       map_location="cpu", weights_only=True)
    assert saved["args"]["model"] == "gpt2", "This exporter is only for the GPT-2 run."
    assert saved["args"]["mode"] == "iso"
    scores = saved["scores"].detach().cpu()
    assert scores.ndim == 1 and torch.isfinite(scores).all()

    add_mib_to_sys_path()
    from eap.graph import Graph

    # Explicit GPT-2-small structural config; no tokenizer or model weights needed.
    config = {"n_layers": 12, "n_heads": 12,
              "parallel_attn_mlp": False, "d_model": 768}
    graph = Graph.from_model(config)
    assert graph.neurons_in_graph is None
    valid = graph.real_edge_mask.bool()
    assert int(valid.sum()) == scores.numel() == 32491, "Graph/score mismatch."
    graph.scores[:] = -torch.inf
    graph.scores[valid] = scores

    coordinates = valid.nonzero(as_tuple=False)
    by_coord = {tuple(int(x) for x in edge.matrix_index): edge
                for edge in graph.edges.values()}
    assert len(by_coord) == scores.numel(), "Edge mapping is not one-to-one."
    assert set(by_coord) == {tuple(x) for x in coordinates.tolist()}

    dense_to_score = torch.full((valid.numel(),), -1, dtype=torch.long)
    dense_to_score[valid.flatten()] = torch.arange(scores.numel())
    # Use the graph's dense-matrix ranking, matching apply_topn even for ties.
    dense_order = torch.argsort(graph.scores.flatten(), descending=True)[:scores.numel()]
    score_order = dense_to_score[dense_order]
    fields = ["rank", "score_index", "edge", "source", "destination",
              "destination_input", "score", "matrix_row", "matrix_column"]
    rows = []
    for rank, index in enumerate(score_order.tolist(), 1):
        row, column = coordinates[index].tolist()
        edge = by_coord[(row, column)]
        assert torch.equal(edge.score, scores[index])
        rows.append({"rank": rank, "score_index": index,
                     "edge": edge.name, "source": edge.parent.name,
                     "destination": edge.child.name,
                     "destination_input": edge.qkv or "",
                     "score": float(scores[index]),
                     "matrix_row": row, "matrix_column": column})

    out = Path(tempfile.mkdtemp(prefix="edge_export_", dir=str(run)))
    write_csv(out / "all_edges_ranked.csv", rows, fields)
    write_csv(out / "top20_edges.csv", rows[:20], fields)
    print("TOP 20 NAMED EDGES")
    for r in rows[:20]:
        print(f"{r['rank']:2d}. [{r['score_index']:5d}] {r['edge']:28s} S={r['score']:+.6f}")

    fractions = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]
    expected = saved["mib_results"]["weighted_edge_counts"]
    assert len(expected) == len(fractions)
    count_rows = []
    for p, old_count in zip(fractions, expected):
        k = int(p * scores.numel())
        graph.apply_topn(k, absolute=False, level="edge", prune=True)
        count = int(graph.in_graph.sum())
        weighted_count = float(graph.weighted_edge_count())
        assert count == weighted_count
        matches = weighted_count == float(old_count)
        count_rows.append({"requested_fraction": p, "nominal_k": k,
                           "pruned_edges": count,
                           "saved_count": float(old_count), "matches_saved": matches})
    write_csv(out / "pruning_check.csv", count_rows, list(count_rows[0]))
    print("\nPRUNING CHECK")
    for r in count_rows:
        print(f"{100*r['requested_fraction']:5.1f}%: top-{r['nominal_k']:5d} -> "
              f"{r['pruned_edges']:5d} edges; matches saved={r['matches_saved']}")
    if not all(r["matches_saved"] for r in count_rows):
        raise RuntimeError("Pruning differs from saved evaluation. Do not assume exported circuits match.")

    for p in [.01, .02]:
        k = int(p * scores.numel())
        graph.apply_topn(k, absolute=False, level="edge", prune=False)
        assert int(graph.in_graph.sum()) == k
        raw_coords = {tuple(rc) for rc in graph.in_graph.nonzero().tolist()}
        raw = [r for r in rows if (r["matrix_row"], r["matrix_column"]) in raw_coords]
        graph.apply_topn(k, absolute=False, level="edge", prune=True)
        kept_coords = {tuple(rc) for rc in graph.in_graph.nonzero().tolist()}
        kept = [r for r in rows if (r["matrix_row"], r["matrix_column"]) in kept_coords]
        dropped = [r for r in raw if (r["matrix_row"], r["matrix_column"]) not in kept_coords]
        assert kept_coords <= raw_coords
        assert len(raw) == k and len(kept) + len(dropped) == k
        tag = f"{int(100*p)}pct"
        write_csv(out / f"{tag}_before_pruning.csv", raw, fields)
        write_csv(out / f"{tag}_after_pruning.csv", kept, fields)
        write_csv(out / f"{tag}_removed_by_pruning.csv", dropped, fields)
        dot = ["digraph circuit {", "  rankdir=LR;"]
        for r in kept:
            label = f"{r['destination_input'] or 'residual'}; S={r['score']:.3f}"
            dot.append(f"  {json.dumps(r['source'])} -> {json.dumps(r['destination'])} "
                       f"[label={json.dumps(label)}];")
        dot.append("}")
        (out / f"{tag}_circuit.dot").write_text("\n".join(dot))
    (out / "graph_prune_implementation.txt").write_text(inspect.getsource(Graph.prune))
    (out / "export_metadata.json").write_text(json.dumps({
        "run_dir": str(run), "structural_config": config,
        "graph_source": inspect.getfile(Graph),
        "note": "Circuit structure export only; no new behavioural evaluation.",
    }, indent=2))
    print("\nSaved exports:", out)
    print("No model weights were loaded and no training was performed.")


if __name__ == "__main__":
    main()
