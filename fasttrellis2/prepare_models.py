"""Download the public TRELLIS.2 weights into a local folder.

The generated pipeline.json is local state and must not be committed: its
conditioning model paths point to this machine's downloaded files.
"""
import argparse
import json
from pathlib import Path


MODEL_REPO = 'microsoft/TRELLIS.2-4B'
DINO_REPO = 'timm/vit_large_patch16_dinov3.lvd1689m'
BIREFNET_REPO = 'ZhengPeng7/BiRefNet'


def split_checkpoint(reference):
    if reference.startswith('ckpts/'):
        return MODEL_REPO, reference
    parts = reference.split('/')
    if len(parts) < 3:
        raise ValueError(f'Unexpected checkpoint reference: {reference}')
    return '/'.join(parts[:2]), '/'.join(parts[2:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='models/trellis2')
    parser.add_argument('--pipeline-json', help='Optional downloaded official pipeline.json')
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.pipeline_json:
        pipeline = json.loads(Path(args.pipeline_json).read_text(encoding='utf8'))
    else:
        official = hf_hub_download(MODEL_REPO, 'pipeline.json')
        pipeline = json.loads(Path(official).read_text(encoding='utf8'))
    if pipeline.get('name') != 'Trellis2ImageTo3DPipeline':
        raise ValueError('Expected the official TRELLIS.2 image-to-3D pipeline')

    model_refs = pipeline['args']['models']
    for name in tuple(model_refs):
        if name.endswith('_1024'):
            del model_refs[name]
    for name, reference in tuple(model_refs.items()):
        repo, filename = split_checkpoint(reference)
        local_root = (output if repo == MODEL_REPO else
                      output / 'external' / repo.replace('/', '__'))
        for suffix in ('.json', '.safetensors'):
            hf_hub_download(repo, filename + suffix, local_dir=local_root)
        model_refs[name] = (local_root / filename).relative_to(output).as_posix()
        print(f'checkpoint ready: {name}', flush=True)

    dino_dir = output / 'conditioning' / 'dinov3'
    dino_file = Path(hf_hub_download(DINO_REPO, 'model.safetensors',
                                     local_dir=dino_dir)).resolve()
    birefnet_dir = output / 'conditioning' / 'birefnet'
    snapshot_download(BIREFNET_REPO, local_dir=birefnet_dir,
                      allow_patterns=['*.py', 'config.json', 'model.safetensors', 'LICENSE'])
    pipeline['args']['image_cond_model'] = {
        'name': 'PublicDinoV3FeatureExtractor',
        'args': {'model_name': str(dino_file)}}
    pipeline['args']['rembg_model'] = {
        'name': 'BiRefNet', 'args': {'model_name': str(birefnet_dir.resolve())}}
    pipeline['args']['low_vram'] = False
    (output / 'pipeline.json').write_text(
        json.dumps(pipeline, indent=2) + '\n', encoding='utf8')
    print(f'local pipeline ready: {output / "pipeline.json"}', flush=True)


if __name__ == '__main__':
    main()
