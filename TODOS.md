- Reduce actual work before parallelizing: frame differencing/motion detection to skip near-duplicate frames

- Batch frames through the detector instead of threading — stack N frames into one tensor, one forward pass. This uses GPU parallelism properly and avoids Python overhead entirely. Usually the biggest win.
- multiprocessing.Pool instead of threading for the CPU-bound parts (this sidesteps the GIL question entirely, and gives real multi-core parallelism). Cost: each process needs its own model loaded into memory (or GPU memory), so 4 processes = 4x model memory. Fine for a small YOLO/Detectron2 model, less fine if it's huge.
- ThreadPoolExecutor with 4 workers as a quick experiment — cheap to try, and given the native-code-heavy pipeline described above, it may just work.
