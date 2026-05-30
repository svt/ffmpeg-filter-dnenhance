# ffmpeg-filter-dnenhance

An FFmpeg audio filter that performs neural dialogue enhancement using
[DeepFilterNet 3][dfn] via its native Rust inference library `libdf`.

## Purpose

Run DeepFilterNet 3 inference inline in an FFmpeg filter graph — no Python,
no LibTorch, no separate subprocess. The filter does one thing only: take
mono dialogue audio, return enhanced mono audio. Surrounding processing
(stereo downmix, channelsplit, sidechain compression) is left to standard
FFmpeg primitives.

This is a downstream patch against FFmpeg, in the same style as
[ffmpeg-filter-proxy][proxy]; it is not intended for upstream FFmpeg.

## Filter contract

| | |
|---|---|
| Filter name | `dnenhance` |
| Input | mono, `AV_SAMPLE_FMT_FLTP`, 48000 Hz |
| Output | mono, `AV_SAMPLE_FMT_FLTP`, 48000 Hz |
| Hop size | 480 samples (10 ms) |
| Algorithmic latency | 20 ms (2 hops of DFN3 lookahead) |
| Linkage | runtime via `dlopen` of `libdf.so` / `libdf.dylib` |

The filter only accepts mono input. To enhance dialogue in a stereo or
multichannel source, surround `dnenhance` with standard channel-routing
filters (`pan`, `channelsplit`, `amerge`, `join`, `sidechaincompress`).
See the [Examples](#examples) below.

## Options

| Option | Type | Default | Description |
|---|---|---|---|
| `model` | string | auto-discover | Path to the DFN3 model tarball. If empty, searches standard Homebrew prefixes for `share/libdf/DeepFilterNet3.tar.gz`. |
| `post_filter` | bool | `1` | Enable DFN3 post-filter (extra suppression refinement). Maps to libdf `beta = 0.02` when on, `0.0` when off. |
| `attenuation_limit` | float (dB) | `100.0` | Maximum suppression in dB. Effectively unlimited at 100. |
| `log_level` | string | `"None"` | libdf log level (`None`, `Error`, `Warn`, `Info`, `Debug`, `Trace`). |
| `lookahead` | int hops | `2` | Model algorithmic lookahead in 480-sample hops. **Must match the model**: DFN3 standard = `2` (20 ms latency), DFN3-LL = `0` (no latency). For other models, read `df_lookahead` and `conv_lookahead` from the tarball's `config.ini` and use the max. |

## Building

The build has two parts: build `libdf` from the DeepFilterNet repository,
then patch and rebuild FFmpeg with the new filter.

### 1. Build `libdf`

```sh
./scripts/build-libdf.sh
```

This clones DeepFilterNet into `./build/DeepFilterNet`, runs
`cargo build --release -p deep_filter --features "capi,default-model,tract"`,
and copies the resulting shared library into `./build/lib/`.

Requires a recent Rust toolchain (`rustup`).

### 2. Patch FFmpeg

```sh
cp af_dnenhance.c <your-ffmpeg-checkout>/libavfilter/
cd <your-ffmpeg-checkout>
```

Add the filter object to `libavfilter/Makefile`:

```make
OBJS-$(CONFIG_DNENHANCE_FILTER) += af_dnenhance.o
```

Register the filter in `libavfilter/allfilters.c`:

```c
extern const FFFilter ff_af_dnenhance;
```

FFmpeg's `configure` script auto-discovers filters by grepping
`libavfilter/allfilters.c` for `extern const FFFilter ff_*;` lines, so
no edit to `configure` or `--enable-filter=` flag is needed — the
declaration above is enough to opt the filter into the default build.

Then build with your usual flags:

```sh
./configure --enable-gpl ...
make -j$(nproc)
```

`libdf` is loaded at runtime, so the FFmpeg build itself does not depend on
having `libdf` installed.

## Usage

At runtime, point the dynamic loader at the `libdf` shared library you
built in step 1 (the script copies it into `./build/lib/` relative to
this repo's root):

```sh
# Linux
LD_LIBRARY_PATH=./build/lib \
  ffmpeg -i in.wav \
    -af "dnenhance=model=/path/to/DeepFilterNet3_onnx.tar.gz" \
    out.wav

# macOS
DYLD_LIBRARY_PATH=./build/lib \
  ffmpeg -i in.wav \
    -af "dnenhance=model=/path/to/DeepFilterNet3_onnx.tar.gz" \
    out.wav
```

Check that the filter is registered:

```sh
ffmpeg -h filter=dnenhance
```

## Model file resolution

The `model=` option points the filter at a DFN3 ONNX tarball
(`DeepFilterNet3_onnx.tar.gz` or similar). It is **optional**: if not
provided, `af_dnenhance` searches the standard Homebrew install prefixes
for a `share/libdf/DeepFilterNet3.tar.gz` file and uses the first one it
finds. The fallback list is:

- `/opt/homebrew/share/libdf/DeepFilterNet3.tar.gz` (macOS Apple Silicon)
- `/usr/local/share/libdf/DeepFilterNet3.tar.gz` (macOS Intel)
- `/home/linuxbrew/.linuxbrew/share/libdf/DeepFilterNet3.tar.gz` (Linuxbrew)

This means if you install `libdf` via the same Homebrew tap that ships
this filter, you don't have to pass `model=` at all:

```sh
brew install svt/avtools/libdf svt/avtools/ffmpeg-encore
ffmpeg -i in.wav -af "dnenhance" out.wav
```

> **Note on `default-model`:** `libdf`'s build also embeds the same DFN3
> tarball into `libdf.dylib` (~8 MB) via cargo's `default-model` feature,
> which is transitively required by the `capi` feature on upstream's
> v0.5.6/main. **That embedded copy is not reachable from the C API** —
> `df_create(path, ...)` always opens the path from disk. So the
> standalone tarball install is what's actually used; the embedded copy
> is dead weight we accept until upstream restructures the feature flags.
> If you build `libdf` yourself outside of Homebrew (or install the
> tarball elsewhere), pass `model=<path>` explicitly.

## Examples

### Mono dialogue file

```sh
ffmpeg -i mono_48k.wav \
  -af "dnenhance=model=/path/to/DeepFilterNet3_onnx.tar.gz" \
  enhanced.wav
```

### Enhance the center of a stereo file (AC-3-style downmix)

```sh
ffmpeg -i stereo_48k.wav \
  -af "pan=mono|c0=0.707*c0+0.707*c1,dnenhance" \
  enhanced_mono.wav
```

### Stereo in, stereo out — with sidechain ducking

This is the production pattern used by `encore`: downmix → enhance →
use the enhanced mono as a sidechain key to duck the background of the
original stereo → layer the enhanced mono back into both channels.
Preserves the stereo image of the music/ambience while making dialogue
louder and cleaner. Listener preference matches the textbook: keep the
source's stereo width, re-center the boosted dialogue.

```sh
ffmpeg -i in.wav -filter_complex "\
[0:a]asplit=2[orig][m];\
[m]pan=mono|c0=0.707*c0+0.707*c1,dnenhance,asplit=2[sc][fc];\
[orig][sc]sidechaincompress=threshold=0.012:ratio=8:attack=100:release=1000[compr];\
[compr][fc]amerge,pan=stereo|c0<c0+c2|c1<c1+c2[out]" \
  -map "[out]" out.wav
```

The `sidechaincompress` knobs (`threshold`, `ratio`, `attack`,
`release`) are perceptual — the values above are the defaults encore
ships. Tune them down if the ducking feels too aggressive, or steepen
them if dialogue still doesn't sit forward enough in busy mixes.

### Enhance the FC channel of a 5.1 file, keep the rest

```sh
ffmpeg -i in_51.wav -filter_complex "
  [0:a]channelsplit=channel_layout=5.1[FL][FR][FC][LFE][BL][BR];
  [FC]dnenhance=model=/path/to/DFN3.tar.gz[FC2];
  [FL][FR][FC2][LFE][BL][BR]join=inputs=6:channel_layout=5.1[out]
" -map "[out]" out_51.wav
```

### Low-latency mode for live transcoding

DeepFilterNet ships a low-latency variant (`DeepFilterNet3_ll_onnx.tar.gz`)
with **0 ms** algorithmic latency, at a modest quality cost. The filter
needs `lookahead=0` to use it correctly — otherwise it will discard the
first two real outputs and emit two trailing frames of enhanced silence.

```sh
ffmpeg -i in.wav \
  -af "dnenhance=model=/path/to/DeepFilterNet3_ll.tar.gz:lookahead=0" \
  out.wav
```

## How it works

DeepFilterNet 3 has a 2-hop algorithmic lookahead: a call to
`df_process_frame(hop_N)` returns the enhanced version of `hop_(N-2)`. The
first two outputs after startup are "enhanced zeros" from the model's
cold-start buffer and are discarded; at end-of-stream the filter pushes two
hops of silence to flush the model and emit the enhanced versions of the
final two real hops.

If the input doesn't end on a hop boundary, the trailing partial hop is
zero-padded to a full hop before being pushed through the model. The
algorithm's behavior on that tail is undefined relative to a hypothetical
infinitely-long input, but the alternative (silently discarding samples) is
worse.

## License

Copyright 2026 Sveriges Television AB.

This software is released under the GNU Lesser General Public License
version 2.1 or later (LGPL v2.1+).

[dfn]: https://github.com/Rikorose/DeepFilterNet
[proxy]: https://github.com/svt/ffmpeg-filter-proxy
