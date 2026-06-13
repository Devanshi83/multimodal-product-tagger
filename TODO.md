# TODO / Known Limitations

## Architecture
- [ ] Train separate classification heads per modality for valid unimodal ablation
- [ ] Add Phase 2 fine-tuning run (top-4 CLIP vision blocks unfrozen, epochs 6-20)
- [ ] Experiment with cross-attention fusion instead of simple concatenation

## Evaluation
- [ ] Deploy FastAPI to Railway/Render with public endpoint
- [ ] Add per-class F1 breakdown for rare articleType classes
- [ ] Run full 20-epoch training for publication-quality numbers

## Engineering
- [ ] Add integration tests for FastAPI endpoints
- [ ] Add pre-commit hooks (ruff, black)
- [ ] Benchmark inference latency (CPU vs MPS vs CUDA)
