"""Build the four native modules and CUDA arena used by the 512 pipeline."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cumesh-source', default='third_party/CuMesh',
                        help='Source checkout of the CuMesh version installed in this environment')
    parser.add_argument('--eigen-include',
                        help='Eigen include directory when the CuMesh source checkout is unavailable')
    parser.add_argument('--output-dir', default='fasttrellis2/build')
    args = parser.parse_args()

    import cumesh._cubvh
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        parser.error('CUDA_HOME is required to compile the native modules')
    repo = Path(__file__).resolve().parents[1]
    native = repo / 'fasttrellis2/native'
    cumesh_source = Path(args.cumesh_source).expanduser().resolve()
    eigen = (Path(args.eigen_include).expanduser().resolve() if args.eigen_include
             else cumesh_source / 'third_party/cubvh/third_party/eigen')
    if not (eigen / 'Eigen/Dense').is_file():
        parser.error(f'Eigen headers are missing: {eigen}')
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '4')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')

    def build(name, filenames, cflags, cuda_flags=None, ldflags=None, includes=None):
        build_dir = output / ('objects_' + name)
        build_dir.mkdir(exist_ok=True)
        module = load(name=name, sources=[str(native / file) for file in filenames],
            build_directory=str(build_dir), extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags or [], extra_ldflags=ldflags or [],
            extra_include_paths=[str(path) for path in (includes or [])], verbose=True)
        destination = output / Path(module.__file__).name
        shutil.copy2(module.__file__, destination)
        print(f'built {name}: {destination}', flush=True)

    xatlas_flags = ['-O3', '-DNDEBUG', '-std=c++17', '-ffp-contract=off',
                    '-mavx512f', '-mavx512dq', '-mavx512vl']
    build('trellis_xatlas_bounded_whole',
          ['binding_bounded_groups.cpp', 'xatlas_boundary_fused_costs.cpp'],
          xatlas_flags + ['-fopenmp'], ldflags=['-fopenmp'])
    telea_cuda = ['-O3', '--fmad=false', '--prec-div=true',
                  '--prec-sqrt=true', '-std=c++17']
    build('trellis_telea_gpu', ['telea_plan.cpp', 'telea_render.cu'],
          ['-O3', '-ffp-contract=off', '-std=c++17'], telea_cuda)
    build('trellis_telea_stream_seeded',
          ['telea_stream_seeded.cpp', 'telea_stream_seeded.cu'],
          ['-O3', '-ffp-contract=off', '-fno-math-errno', '-march=native',
           '-std=c++17'], telea_cuda)
    build('trellis_bvh_async', ['bvh_async_binding.cpp'],
          ['-O3', '-DNDEBUG', '-std=c++17'],
          ldflags=[cumesh._cubvh.__file__], includes=[eigen])

    cuda_lib = Path(CUDA_HOME) / 'lib64'
    if not cuda_lib.is_dir():
        cuda_lib = Path(CUDA_HOME) / 'targets/x86_64-linux/lib'
    if not cuda_lib.is_dir():
        parser.error(f'CUDA runtime library directory is missing below {CUDA_HOME}')
    compiler = os.environ.get('CXX') or shutil.which('c++')
    if not compiler:
        parser.error('A C++ compiler is required for the CUDA arena')
    arena = output / 'libtrellis_cuda_arena.so'
    subprocess.run([compiler, '-shared', '-fPIC', '-O3', '-std=c++17',
        '-I' + str(Path(CUDA_HOME) / 'include'),
        str(native / 'cuda_scratch_arena.cpp'), '-L' + str(cuda_lib),
        '-Wl,-rpath,' + str(cuda_lib), '-lcudart', '-ldl', '-o', str(arena)],
        check=True)
    print(f'built CUDA arena: {arena}', flush=True)


if __name__ == '__main__':
    main()
