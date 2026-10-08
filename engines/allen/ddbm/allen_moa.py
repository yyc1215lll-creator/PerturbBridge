"""Frozen real-only Allen MoA heads from job17424206; never train on generated data."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

from .auxiliary_losses import sha256_file

INPUT_SHA = '6445a94abe21d2f48ccdd04dc4300fbc58145577a1814f78a54103f5244d4285'
HEAD_SHA = {
    42: '315f7d28a8998a0fb6944148e129e7c02becf90096d894b962252b8f997191ea',
    43: '64d45d2934bb155b5d4820bf94b222bdff2823d0fc16c0fcdc65f3bb7ebf3cbe',
    44: 'a885d831fda6da2fa745fcc3a227df92633b101ea1f6d92a50924f9b7f7807d6',
}


def measures(prob, labels):
    assert prob.shape == (len(labels), 5) and np.isfinite(prob).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-5)
    pred = prob.argmax(1)
    return dict(accuracy=float(accuracy_score(labels, pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                macro_f1=float(f1_score(labels, pred, labels=range(5), average='macro', zero_division=0)),
                weighted_f1=float(f1_score(labels, pred, labels=range(5), average='weighted', zero_division=0)),
                confusion_matrix=confusion_matrix(labels, pred, labels=range(5)).tolist(), count=len(labels))


def validate_inputs(manifest_path, head_root):
    assert sha256_file(manifest_path) == INPUT_SHA
    m = json.loads(Path(manifest_path).read_text())
    assert m['seeds'] == [42, 43, 44]
    assert sha256_file(m['weights']) == m['weights_sha256']
    for seed, digest in HEAD_SHA.items():
        assert sha256_file(Path(head_root)/f'head-{seed}.pth') == digest
    return m


@torch.inference_mode()
def score_images(image_root, output, manifest_path, head_root, device):
    m = validate_inputs(manifest_path, head_root)
    pop = m['populations']['real']
    paths = list(Path(image_root).rglob('*.png'))
    lookup = {p.stem: p for p in paths}
    assert len(paths) == len(lookup) == len(pop['keys']) == 1592
    assert set(lookup) == set(pop['keys'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        encoder = FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
            feature_extractor_weights_path=m['weights']).to(device).eval().requires_grad_(False)
        features = []
        for i in range(0, len(pop['keys']), 32):
            pixels = []
            for key in pop['keys'][i:i+32]:
                with Image.open(lookup[key]) as im:
                    pixels.append(torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1))
            features.append(encoder(torch.stack(pixels).to(device))[0].cpu().numpy())
        f = np.concatenate(features)
        del encoder
        y = np.asarray([m['vocab'].index(r['ANNOT']) for r in pop['rows']])
        structures = np.asarray([r['STRUCTURE'] for r in pop['rows']])
        results = {}
        for seed in m['seeds']:
            head = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.ReLU(),
                                      torch.nn.Dropout(.5), torch.nn.Linear(512, 5)).to(device).eval()
            state = torch.load(Path(head_root)/f'head-{seed}.pth', map_location='cpu', weights_only=False)
            head.load_state_dict(state['model'], strict=True)
            prob = np.concatenate([head(torch.from_numpy(f[i:i+256]).to(device)).softmax(1).cpu().numpy()
                                   for i in range(0, len(f), 256)])
            results[str(seed)] = dict(cell=measures(prob, y),
                per_structure={s: measures(prob[structures == s], y[structures == s]) for s in sorted(set(structures))})
            np.savez_compressed(output.with_suffix(f'.seed{seed}.npz'), probabilities=prob,
                                sample_keys=np.asarray(pop['keys']))
            del head
        np.savez_compressed(output.with_suffix('.features.npz'), features=f, sample_keys=np.asarray(pop['keys']))
        summary = {key: float(np.mean([v['cell'][key] for v in results.values()]))
                   for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1')}
        summary.update(total=len(y), seeds=results, head_sha256=HEAD_SHA,
                       input_manifest_sha256=INPUT_SHA, primary_metric='accuracy',
                       protocol=m['protocol'], image_root=str(image_root), frozen_heads=True)
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
        temp.replace(output)
        print(f"ALLEN_FIXED_MOA_SUCCESS accuracy={summary['accuracy']:.8f} n={len(y)}", flush=True)
        return summary
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prior_tf32
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
