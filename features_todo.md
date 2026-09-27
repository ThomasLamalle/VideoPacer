List of features to implement. Increasing numbers does not mean features depends on each other, assume independance first.
X features needs brainstorming or analyze before consideration

- Feature 1: Reduce actual work before parallelizing: frame differencing/motion detection to skip near-duplicate frames

- Feature 2: Batch frames through the detector instead of threading — stack N frames into one tensor, one forward pass. This uses GPU parallelism properly and avoids Python overhead entirely. Usually the biggest win.
- Feature 3: multiprocessing.Pool instead of threading for the CPU-bound parts (this sidesteps the GIL question entirely, and gives real multi-core parallelism). Cost: each process needs its own model loaded into memory (or GPU memory), so 4 processes = 4x model memory. Fine for a small YOLO/Detectron2 model, less fine if it's huge.
- Feature 4 :ThreadPoolExecutor with 4 workers as a quick experiment — cheap to try, and given the native-code-heavy pipeline described above, it may just work.


-Feature 5: Keep track of bibs so we can :
    1) count votes on bibs
    2) improve bib detection ? like if we lower threshold on frame where we did not detect a bib ?
    3) Do less bib detection ? like one every X frames and between the frames we only do tracking ?

- Feature 6 (depends on 5) Keep track of partially read bib and combine results to give a better results. we need to define bib string patterns.


- Feature 7 (depends on 5) Add a locking mechanism on bibs to avoid rereading the bib on each frame. For exmaple, if the bib box has already 5 reads with the same numbers, don't bother read it. For that we need to keep track of the bibs. Maybe Use a dict or set to keep track of the readings

- Feature X : If bib detection is good enough but not reading, we can keep a screenshot of the face for manual review. (If detection is failing it is more problematic as it could be from a organizer or spectator)

- Feature X : For now we do "zero shot" detection, i.e. bib detection without any information on bibs. But before a race we should have access to what the bibs would look like (if it has logos or else) or we could even use already detected bib information to help detection ?
