# Third-party source notices

FastTrellis2 retains Microsoft's TRELLIS.2 code and its root [MIT license](../LICENSE).

- `native/bvh_async_binding.cpp` calls the installed [CuMesh](https://github.com/JeffreyXiang/CuMesh) cuBVH implementation. CuMesh is MIT licensed; its notice is in [native/LICENSE-CuMesh](native/LICENSE-CuMesh). The build instructions use CuMesh commit `12289e1062f0603f2f0d0771b02e1395d247f26f`.
- `native/xatlas.h` and `native/xatlas_boundary_fused_costs.cpp` derive from [xatlas](https://github.com/jpcy/xatlas). Its MIT notice and the included Fast-BVH notice remain at the top of the source files. `native/binding_bounded_groups.cpp` supplies the bounded connected grouping and Python binding.
- `native/telea_plan.cpp` and `native/telea_stream_seeded.cpp` adapt OpenCV's Telea inpainting implementation. The Intel Open Source Computer Vision Library license notice remains at the top of both files. Their CUDA render counterparts implement the corresponding pixel calculation.
- [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), [nvdiffrast](https://github.com/NVlabs/nvdiffrast), and the public model weights are installed separately and are not copied into this repository.

The model files downloaded by `prepare_models.py` are excluded from Git. Review the model repositories' own terms before redistributing weights or generated assets.
