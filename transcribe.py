"""Clip -> timed speech segments with faster-whisper.

Timestamps come ONLY from here. The language model never sees the clock,
so it is never allowed to produce times.
"""
import json
import os
import pathlib
import sys


def _add_cuda_dlls():
    """Windows + CUDA only. faster-whisper (CTranslate2) needs cuBLAS 12 and cuDNN 9.
    If they were installed with pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) the DLLs sit in
    site-packages/nvidia/<lib>/bin, which is not on PATH. Add those folders so they load."""
    if os.name != "nt":
        return
    try:
        import nvidia  # namespace package created by the nvidia-* pip wheels
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in pathlib.Path(base).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def transcribe(clip_path, model_name="small.en", device="cpu", compute_type="int8"):
    """Return [{start, end, text}] and save it to store/transcript_<stem>.json."""
    if device == "cuda":
        _add_cuda_dlls()
    from faster_whisper import WhisperModel  # imported here so DLL paths are set first

    clip = pathlib.Path(clip_path)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    # vad_filter skips silence, so the recognizer only processes speech.
    segments, _info = model.transcribe(str(clip), vad_filter=True)
    # segments is a generator: the model actually runs while we loop over it.
    out = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
           for s in segments if s.text.strip()]
    store = pathlib.Path("store")
    store.mkdir(exist_ok=True)
    (store / f"transcript_{clip.stem}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    # Smoke test:  python transcribe.py data/clips/test.mp4
    import config as cfg
    for s in transcribe(sys.argv[1], cfg.WHISPER, cfg.DEVICE, cfg.COMPUTE):
        print(f"{s['start']:7.1f} {s['end']:7.1f}  {s['text']}")
