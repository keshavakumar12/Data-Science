import sys
from pathlib import Path

from mashup_core import create_mashup


USAGE = (
    "Usage: python 102303982.py <SingerName> <NumberOfVideos> <AudioDuration> <OutputFileName>\n"
    'Example: python 102303982.py "Sharry Maan" 20 25 output.mp3'
)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if len(args) != 4:
        print("Incorrect number of parameters.\n" + USAGE)
        return 1

    singer, n_videos_raw, duration_raw, output_name = args

    try:
        mashup_path = create_mashup(
            singer=singer,
            n_videos=int(n_videos_raw),
            clip_seconds=int(duration_raw),
            output_path=Path(output_name),
        )
    except Exception as exc:  # catch and show any user-facing errors
        print(f"Error: {exc}")
        return 1

    print(f"Mashup created at {mashup_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
