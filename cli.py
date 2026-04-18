"""
cli.py — Command-line interface for VoiceForge
================================================
Usage:
    python cli.py --source source.wav --target target.wav --output out.wav
    python cli.py --source source.wav --target target.wav --pitch -3 --formant 0.9
"""

import argparse
import sys
from voice_converter import VoiceConverter


def main():
    parser = argparse.ArgumentParser(
        description="VoiceForge CLI — convert source voice into target speaker style"
    )
    parser.add_argument("--source",  required=True, help="Source audio file path")
    parser.add_argument("--target",  required=True, help="Target speaker audio file path")
    parser.add_argument("--output",  default="output.wav", help="Output file path (default: output.wav)")
    parser.add_argument("--pitch",   type=int,   default=0,   help="Extra pitch shift in semitones (default: 0)")
    parser.add_argument("--formant", type=float, default=1.0, help="Formant scale factor (default: 1.0)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════╗
║         VoiceForge CLI               ║
╠══════════════════════════════════════╣
║  Source  : {args.source[:38]:<38} ║
║  Target  : {args.target[:38]:<38} ║
║  Output  : {args.output[:38]:<38} ║
║  Pitch   : {str(args.pitch):<38} ║
║  Formant : {str(args.formant):<38} ║
╚══════════════════════════════════════╝
""")

    vc = VoiceConverter()
    result = vc.convert(
        source_path   = args.source,
        target_path   = args.target,
        output_path   = args.output,
        pitch_shift   = args.pitch,
        formant_shift = args.formant,
    )

    stats = result.get("stats", {})
    print("\n📊 Conversion Stats")
    print("─" * 36)
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print(f"\n✅  Saved to: {args.output}")


if __name__ == "__main__":
    main()
