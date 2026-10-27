"""Train on TRAIN masks; evaluate on privileged VAL masks. Never opens test."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from apex import data as D
from .selector import ObjectSelector, batch, loss


@torch.inference_mode()
def evaluate(model, rows, device):
    model.eval()
    result = []
    for start in range(0, len(rows), 64):
        group = rows[start:start+64]
        inputs, labels = batch(group, device)
        out = model(*inputs)
        ranking = out['scores'].argsort(-1, descending=True).cpu().numpy()
        apred = out['audio_class'].argmax(-1).cpu().numpy()
        gpred = out['geom_class'].argmax(-1).cpu().numpy()
        for j, r in enumerate(group):
            result.append(dict(rel=r['rel'], correct=bool(ranking[j, 0] == r['target']),
                               top5=bool(r['target'] in ranking[j, :5]),
                               audio_correct=bool(apred[j] == r['source']),
                               source_geometry_correct=bool(gpred[j, r['target']] == r['source'])))
    return dict(n=len(result), selection_accuracy=float(np.mean([r['correct'] for r in result])),
                top5_accuracy=float(np.mean([r['top5'] for r in result])),
                audio_category_accuracy=float(np.mean([r['audio_correct'] for r in result])),
                source_geometry_category_accuracy=float(np.mean([r['source_geometry_correct'] for r in result])),
                privileged_perfect_masks=True, deployable_segmentation_result=False, samples=result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'best.pt').exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tr = torch.load(args.cache / 'perfect_objects_train.pt', weights_only=False)
    va = torch.load(args.cache / 'perfect_objects_val.pt', weights_only=False)
    assert not ({r['rel'].split('/')[1] for r in tr} & {r['rel'].split('/')[1] for r in va})
    config = {**vars(args), 'cache': str(args.cache), 'output': str(args.output), 'test_used': False,
              'train_samples': len(tr), 'val_samples': len(va), 'privileged_evaluation': True,
              'source_hashes': {f.name: D.sha(f) for f in Path(__file__).parent.glob('*.py')}}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    model = ObjectSelector().to(args.device)
    all_geometry = np.concatenate([r['geometry'] for r in tr])
    model.mean.copy_(torch.as_tensor(all_geometry.mean(0), device=args.device))
    model.std.copy_(torch.as_tensor(all_geometry.std(0).clip(.05), device=args.device))
    freq = np.bincount(np.concatenate([r['categories'] for r in tr]), minlength=17).clip(1)
    weights = np.sqrt(freq.sum() / freq).clip(1, 10)
    weights /= weights.mean()
    weights = torch.as_tensor(weights, dtype=torch.float32, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr*.1)
    rng = np.random.default_rng(args.seed)
    history, best, start = [], -1, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        total = 0
        for b in range(0, len(tr), 32):
            rows = [tr[i] for i in order[b:b+32]]
            inp, lab = batch(rows, args.device, augment=True)
            opt.zero_grad(set_to_none=True)
            value = loss(model(*inp), lab, weights)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite selector loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            total += float(value.detach()) * len(rows)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            report = evaluate(model, va, args.device)
            train_report = evaluate(model, tr[::8], args.device)
            row = dict(epoch=epoch, train_loss=total/len(tr), train_accuracy=train_report['selection_accuracy'],
                       validation_accuracy=report['selection_accuracy'], audio_accuracy=report['audio_category_accuracy'],
                       geometry_accuracy=report['source_geometry_category_accuracy'], elapsed_seconds=time.time()-start)
            history.append(row); print(json.dumps(row), flush=True)
            if report['selection_accuracy'] > best:
                best = report['selection_accuracy']
                torch.save(dict(model=model.state_dict(), config=config, epoch=epoch), args.output / 'best.pt')
                (args.output / 'perfect_mask_validation.json').write_text(json.dumps(report, indent=2))
            (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    (args.output / 'COMPLETE.json').write_text(json.dumps(dict(epochs=args.epochs, test_used=False,
        best_perfect_mask_validation_selection_accuracy=best, checkpoint_sha256=D.sha(args.output/'best.pt')),
        indent=2))


if __name__ == '__main__':
    main()
