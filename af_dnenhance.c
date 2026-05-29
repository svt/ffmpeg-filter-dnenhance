/*
 * SPDX-FileCopyrightText: 2026 Sveriges Television AB
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "audio.h"
#include "filters.h"
#include "formats.h"

// Models like DFN3 have algorithmic lookahead (max of conv_lookahead and
// df_lookahead in libDF's DfTract). df_process_frame returns the enhanced
// version of a hop received `lookahead` calls ago; the first `lookahead`
// outputs after init are "enhanced zeros" and must be discarded, and at
// EOF we push `lookahead` hops of silence to drain real audio that would
// otherwise stay locked in the model's internal buffers.
//
// DFN3 standard: lookahead=2 (20 ms latency).
// DFN3-LL:       lookahead=0 (no latency, lower quality).
// Other models:  read `df_lookahead` and `conv_lookahead` from the model's
//                config.ini and pick max(df_lookahead, conv_lookahead).
//
// The value is per-instance (an AVOption), not a #define, because callers
// may swap models without recompiling.

// Upper bound for the PTS ring. Must be > the max supported lookahead.
#define DFN_PTS_QUEUE 16
#define DFN_LOOKAHEAD_MAX 8

// DFN3 default post-filter strength (Valin/Schmidt et al.). 0.0 disables.
#define POST_FILTER_BETA_ON 0.02f

typedef struct DnEnhanceContext {
    const AVClass *class;

    // options
    char  *model_path;
    int    post_filter;        // bool 0/1
    float  attenuation_limit;  // dB; 100 is effectively unlimited
    char  *log_level;          // libdf log level string
    int    lookahead_hops;     // priming-discard + EOF-drain count

    // libdf dynamic linkage
    void  *libdf_handle;
    void *(*df_create)(const char *, float, const char *);
    size_t (*df_get_frame_length)(void *);
    float  (*df_process_frame)(void *, float *, float *);
    void   (*df_set_atten_lim)(void *, float);
    void   (*df_set_post_filter_beta)(void *, float);
    void   (*df_free)(void *);

    void  *df_state;
    int    hop;                 // libdf-reported frame length (480 at 48 kHz)

    // PTS queue: ring of pending input PTS values waiting for their
    // delayed enhanced output.
    int64_t pts_queue[DFN_PTS_QUEUE];
    int     pts_head;
    int     pts_count;
    int     primed;             // discarded lookahead startup hops?

    // EOF drain state
    int     draining;
    int     drain_remaining;    // hops of silence still to push
    int64_t eof_pts;
} DnEnhanceContext;

#define OFFSET(x) offsetof(DnEnhanceContext, x)
#define AF AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM

static const AVOption dnenhance_options[] = {
    { "model",
      "path to DeepFilterNet 3 model (.tar.gz)",
      OFFSET(model_path), AV_OPT_TYPE_STRING, {.str = NULL}, 0, 0, AF },
    { "post_filter",
      "enable DFN3 post-filter (extra suppression refinement)",
      OFFSET(post_filter), AV_OPT_TYPE_BOOL,  {.i64 = 1}, 0, 1, AF },
    { "attenuation_limit",
      "max suppression in dB (100 = effectively unlimited)",
      OFFSET(attenuation_limit), AV_OPT_TYPE_FLOAT,
      {.dbl = 100.0}, 0.0, 200.0, AF },
    { "log_level",
      "libdf log level (None, Error, Warn, Info, Debug, Trace)",
      OFFSET(log_level), AV_OPT_TYPE_STRING, {.str = "None"}, 0, 0, AF },
    { "lookahead",
      "model lookahead in 480-sample hops (DFN3: 2, DFN3-LL: 0)",
      OFFSET(lookahead_hops), AV_OPT_TYPE_INT,
      {.i64 = 2}, 0, DFN_LOOKAHEAD_MAX, AF },
    { NULL }
};

AVFILTER_DEFINE_CLASS(dnenhance);

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = {
        AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE
    };
    int sample_rates[] = { 48000, -1 };
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    AVFilterChannelLayouts *layouts = NULL;
    int ret;

    if ((ret = ff_set_common_formats_from_list2(ctx, cfg_in, cfg_out,
                                                sample_fmts)) < 0)
        return ret;
    if ((ret = ff_set_common_samplerates_from_list2(ctx, cfg_in, cfg_out,
                                                    sample_rates)) < 0)
        return ret;
    if ((ret = ff_add_channel_layout(&layouts, &mono)) < 0)
        return ret;
    return ff_set_common_channel_layouts2(ctx, cfg_in, cfg_out, layouts);
}

static int load_sym(AVFilterContext *ctx, void *handle, const char *name,
                    void **out)
{
    *out = dlsym(handle, name);
    if (!*out) {
        av_log(ctx, AV_LOG_ERROR, "libdf missing symbol %s: %s\n",
               name, dlerror());
        return AVERROR(EINVAL);
    }
    return 0;
}

static int open_libdf(AVFilterContext *ctx)
{
    DnEnhanceContext *s = ctx->priv;
    static const char *const candidates[] = {
        // bare names — resolved via LD_LIBRARY_PATH / DYLD_LIBRARY_PATH /
        // system search path
        "libdf.so",
        "libdf.dylib",
        "libdf.so.0",
        // common Homebrew install prefixes so a brew-installed libdf works
        // without any environment variables
        "/opt/homebrew/lib/libdf.dylib",
        "/usr/local/lib/libdf.dylib",
        "/opt/homebrew/lib/libdf.so",
        "/usr/local/lib/libdf.so",
        "/home/linuxbrew/.linuxbrew/lib/libdf.so",
        NULL,
    };
    for (int i = 0; candidates[i]; i++) {
        s->libdf_handle = dlopen(candidates[i], RTLD_LAZY | RTLD_LOCAL);
        if (s->libdf_handle)
            break;
    }
    if (!s->libdf_handle) {
        const char *err = dlerror();
        av_log(ctx, AV_LOG_ERROR,
               "cannot load libdf (set LD_LIBRARY_PATH or DYLD_LIBRARY_PATH): %s\n",
               err ? err : "unknown");
        return AVERROR(EINVAL);
    }

    int ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_create",
                        (void **)&s->df_create)) < 0) return ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_get_frame_length",
                        (void **)&s->df_get_frame_length)) < 0) return ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_process_frame",
                        (void **)&s->df_process_frame)) < 0) return ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_set_atten_lim",
                        (void **)&s->df_set_atten_lim)) < 0) return ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_set_post_filter_beta",
                        (void **)&s->df_set_post_filter_beta)) < 0) return ret;
    if ((ret = load_sym(ctx, s->libdf_handle, "df_free",
                        (void **)&s->df_free)) < 0) return ret;
    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    DnEnhanceContext *s = ctx->priv;
    int ret;

    // Resolve the model path. If the user didn't pass one, fall back to the
    // standard brew install location of libdf's DFN3 tarball. Lets a brew-
    // installed libdf "just work" without anyone typing a path.
    const char *model_path = s->model_path;
    if (!model_path || !*model_path) {
        static const char *const default_model_paths[] = {
            "/opt/homebrew/share/libdf/DeepFilterNet3.tar.gz",
            "/usr/local/share/libdf/DeepFilterNet3.tar.gz",
            "/home/linuxbrew/.linuxbrew/share/libdf/DeepFilterNet3.tar.gz",
            NULL,
        };
        for (int i = 0; default_model_paths[i]; i++) {
            if (access(default_model_paths[i], R_OK) == 0) {
                model_path = default_model_paths[i];
                av_log(ctx, AV_LOG_VERBOSE,
                       "dnenhance: model option not set, using %s\n", model_path);
                break;
            }
        }
        if (!model_path || !*model_path) {
            av_log(ctx, AV_LOG_ERROR,
                   "the 'model' option is required (no DFN3 tarball found at "
                   "any standard brew prefix; install libdf or pass model=<path>)\n");
            return AVERROR(EINVAL);
        }
    }

    if ((ret = open_libdf(ctx)) < 0)
        return ret;

    // libdf's log_level parser accepts off/error/warn/info/debug/trace, or
    // NULL to disable. Translate "None"/empty (the convention used by the
    // Python wrapper) to NULL so log_level=None disables logging cleanly.
    const char *libdf_log_level = s->log_level;
    if (libdf_log_level &&
        (*libdf_log_level == '\0' || strcasecmp(libdf_log_level, "None") == 0)) {
        libdf_log_level = NULL;
    }
    s->df_state = s->df_create(model_path, s->attenuation_limit,
                               libdf_log_level);
    if (!s->df_state) {
        av_log(ctx, AV_LOG_ERROR,
               "df_create failed for model '%s'\n", model_path);
        return AVERROR_EXTERNAL;
    }
    s->df_set_post_filter_beta(s->df_state,
                               s->post_filter ? POST_FILTER_BETA_ON : 0.0f);

    s->hop = (int)s->df_get_frame_length(s->df_state);
    if (s->hop <= 0) {
        av_log(ctx, AV_LOG_ERROR,
               "df_get_frame_length returned %d\n", s->hop);
        return AVERROR_EXTERNAL;
    }

    av_log(ctx, AV_LOG_VERBOSE,
           "dnenhance: model=%s hop=%d lookahead=%d post_filter=%d atten_lim=%.1f dB\n",
           model_path, s->hop, s->lookahead_hops, s->post_filter, s->attenuation_limit);
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    DnEnhanceContext *s = ctx->priv;
    if (s->df_state && s->df_free) {
        s->df_free(s->df_state);
        s->df_state = NULL;
    }
    if (s->libdf_handle) {
        dlclose(s->libdf_handle);
        s->libdf_handle = NULL;
    }
}

// Push exactly `hop` samples through libdf. Allocates and (after priming)
// emits one delayed output frame.
static int push_hop(AVFilterContext *ctx, const float *in_samples,
                    int64_t pts_in)
{
    DnEnhanceContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;

    out = ff_get_audio_buffer(outlink, s->hop);
    if (!out)
        return AVERROR(ENOMEM);

    s->df_process_frame(s->df_state,
                        (float *)in_samples,
                        (float *)out->extended_data[0]);

    if (s->pts_count < DFN_PTS_QUEUE) {
        s->pts_queue[(s->pts_head + s->pts_count) % DFN_PTS_QUEUE] = pts_in;
        s->pts_count++;
    }

    if (!s->primed) {
        if (s->pts_count <= s->lookahead_hops) {
            // Output is enhanced zeros from the model's cold-start buffer.
            av_frame_free(&out);
            return 0;
        }
        s->primed = 1;
    }

    out->pts = s->pts_queue[s->pts_head];
    s->pts_head = (s->pts_head + 1) % DFN_PTS_QUEUE;
    s->pts_count--;

    return ff_filter_frame(outlink, out);
}

static int push_hop_from_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    const float *samples = (const float *)in->extended_data[0];
    int64_t pts = in->pts;
    int ret = push_hop(ctx, samples, pts);
    av_frame_free(&in);
    return ret;
}

static int drain_one_zero_hop(AVFilterContext *ctx)
{
    DnEnhanceContext *s = ctx->priv;
    float *zeros = av_calloc(s->hop, sizeof(float));
    int ret;
    if (!zeros)
        return AVERROR(ENOMEM);
    ret = push_hop(ctx, zeros, s->eof_pts);
    av_free(zeros);
    return ret;
}

static int handle_eof(AVFilterContext *ctx, int64_t pts)
{
    DnEnhanceContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];

    s->eof_pts = pts;
    s->draining = 1;

    // Drain any sub-hop tail by zero-padding to a full hop.
    int tail = ff_inlink_queued_samples(inlink);
    if (tail > 0) {
        AVFrame *partial = NULL;
        int ret = ff_inlink_consume_samples(inlink, 1, tail, &partial);
        if (ret > 0 && partial) {
            float *scratch = av_calloc(s->hop, sizeof(float));
            if (!scratch) {
                av_frame_free(&partial);
                return AVERROR(ENOMEM);
            }
            memcpy(scratch, partial->extended_data[0],
                   partial->nb_samples * sizeof(float));
            int64_t partial_pts = partial->pts;
            av_frame_free(&partial);
            ret = push_hop(ctx, scratch, partial_pts);
            av_free(scratch);
            if (ret < 0)
                return ret;
        }
    }

    s->drain_remaining = s->lookahead_hops;
    ff_filter_set_ready(ctx, 100);
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    DnEnhanceContext *s = ctx->priv;
    AVFrame *in = NULL;
    int ret, status;
    int64_t pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    if (s->draining) {
        if (s->drain_remaining > 0) {
            ret = drain_one_zero_hop(ctx);
            if (ret < 0)
                return ret;
            s->drain_remaining--;
            ff_filter_set_ready(ctx, 100);
            return 0;
        }
        ff_outlink_set_status(outlink, AVERROR_EOF, s->eof_pts);
        return 0;
    }

    ret = ff_inlink_consume_samples(inlink, s->hop, s->hop, &in);
    if (ret < 0)
        return ret;
    if (ret > 0)
        return push_hop_from_frame(inlink, in);

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF)
            return handle_eof(ctx, pts);
        ff_outlink_set_status(outlink, status, pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);
    return FFERROR_NOT_READY;
}

static const AVFilterPad inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_AUDIO,
    },
};

const FFFilter ff_af_dnenhance = {
    .p.name        = "dnenhance",
    .p.description = NULL_IF_CONFIG_SMALL(
        "Neural dialogue enhancement (DeepFilterNet 3, via libdf)."),
    .p.priv_class  = &dnenhance_class,
    .priv_size     = sizeof(DnEnhanceContext),
    .init          = init,
    .uninit        = uninit,
    .activate      = activate,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
