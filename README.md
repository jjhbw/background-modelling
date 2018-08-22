# GMM Background Subtraction

Code to accompany [my post](https://barkeywolf.consulting/posts/background-subtraction/) on using Gaussian Mixture Modelling for the background subtraction computer vision task (the Stauffer-Grimson / Zivkovic way).

Tested on Rust 1.28 `stable-x86_64-apple-darwin`, `rustc 1.28.0 (9634041f0 2018-07-30)`.



# TODO

- [ ] Fix the bug that somehow increases runtime of prediction performance benchmarks to infeasible durations while the actual time taken by individual iterations is minimal.
- [ ] Maybe implement the improved heuristic proposed by Eric Thul

```
  Adaptive Background Mixture Models for Real-Time Tracking† 
  Eric Thul
  †By Chris Stauffer and W.E.L Grimson [4, 1] 
  March 30, 2007
```



# Testing with example data

**Using your own static video:**

First, we need to cut up the mp4 frames into png images. Note that we take the first number of seconds with `-t` and downsample the fps with `-r`.

```bash
ffmpeg \
	-i ./example_data/traffic/traffic.mp4 \
    -t 20 \
	-r 10 \
	./example_data/traffic/img%4d.png
```

The `-tr` argument sorts the test data by date modified in reverse (so oldest first).

Next, we can train/predict on frames of this dataset:

```bash
cargo run --release $(ls -tr ./example_data/traffic/*.png)
```



**On the 'academic' examples**

Downloaded from [here](http://sbmi2015.na.icar.cnr.it/SBIdataset.html):

```bash
cargo run --release $(ls ./example_data/HighwayI/*.png)
```

```bash
cargo run --release $(ls ./example_data/HighwayII/*.png)
```

These commands produce files named `testimg_predicted_*.png` in the working directory.



### Combine predicted binary mask frames back into videos
The output images generated with the above commands can be combined into a video using:
```bash
ffmpeg -framerate 30 -i ./testimg_predicted_%4d.png -pix_fmt yuv420p segmentations.mp4
```

Note the `-framerate` argument. 

The `%4d` wildcard expects leading-zero numbering like `0001`, `0002`, etc.

**A note for MacOS**: to make an `.mp4` file compatible with macOS, add the `-pix_fmt yuv420p` flag.



### Render two videos next to each other

See this [StackExchange post.](https://unix.stackexchange.com/a/233833)

```bash
ffmpeg \
  -i inputs.mp4 \
  -i segmentations.mp4 \
  -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
  -map [vid] \
  -c:v libx264 \
  -crf 23 \
  -preset veryfast \
  output.mp4
```



# Performance benchmarking

- Be sure to first kill all other programs on the system that may interfere
- Note that we force Rayon to use only a single thread using an environment variable

```bash
RAYON_NUM_THREADS=1 cargo bench
```

To only run a subset of available benchmarks, use a substring of the relevant benchmark names:

```bash
RAYON_NUM_THREADS=1 cargo bench -- model_update
```

