#!/usr/bin/env python3
"""
Generate audio files from transcripts using gTTS (Google Text-to-Speech).
Uses en-US voice as gTTS doesn't have native Singlish support, but will pronounce the text reasonably.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import argparse

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not installed. Run: pip install gTTS")


def generate_audio_gtts(text: str, output_path: Path, lang: str = "en", tld: str = "com.sg") -> bool:
    """
    Generate audio file using gTTS.

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        lang: Language code (default: en)
        tld: Top-level domain for accent (com.sg for Singapore English)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use Singapore English accent by setting tld='com.sg'
        # This gives a closer approximation to Singlish pronunciation
        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)

        # Save as MP3 first
        mp3_path = output_path.with_suffix('.mp3')
        tts.save(str(mp3_path))

        # Convert MP3 to WAV using ffmpeg
        import subprocess
        wav_path = output_path.with_suffix('.WAV')
        subprocess.run([
            'ffmpeg', '-i', str(mp3_path),
            '-ar', '16000',  # 16kHz sample rate (standard for ASR)
            '-ac', '1',       # mono
            '-y',             # overwrite
            str(wav_path)
        ], check=True, capture_output=True)

        # Remove temporary MP3
        mp3_path.unlink()

        return True
    except Exception as e:
        print(f"Error generating audio for {output_path}: {e}")
        return False


def load_transcripts(json_path: Path) -> List[Dict]:
    """Load transcripts from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_all_audio(
    transcripts_path: Path,
    output_dir: Path,
    force: bool = False
) -> Dict[str, int]:
    """
    Generate audio files for all transcripts.

    Args:
        transcripts_path: Path to transcripts JSON file
        output_dir: Directory to save audio files
        force: If True, regenerate even if files exist

    Returns:
        Dictionary with counts: {success: N, skipped: M, failed: K}
    """
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")

    # Load transcripts
    transcripts = load_transcripts(transcripts_path)
    print(f"Loaded {len(transcripts)} transcripts from {transcripts_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate audio for each transcript
    stats = {"success": 0, "skipped": 0, "failed": 0}

    for i, item in enumerate(transcripts, 1):
        filename = item['file']
        transcript = item['transcript']
        category = item['category']

        output_path = output_dir / filename

        # Skip if file exists and not forcing regeneration
        if output_path.exists() and not force:
            print(f"[{i}/{len(transcripts)}] Skipping {filename} (already exists)")
            stats['skipped'] += 1
            continue

        print(f"[{i}/{len(transcripts)}] Generating {filename} ({category}): {transcript[:50]}...")

        success = generate_audio_gtts(transcript, output_path)

        if success:
            stats['success'] += 1
            # Verify file was created
            if output_path.exists():
                file_size = output_path.stat().st_size
                print(f"  ✓ Created {filename} ({file_size} bytes)")
            else:
                print(f"  ✗ Failed to create {filename}")
                stats['failed'] += 1
                stats['success'] -= 1
        else:
            stats['failed'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio files from transcripts using gTTS"
    )
    parser.add_argument(
        '--transcripts',
        type=Path,
        default=Path('data/guardrails/transcripts_realistic.json'),
        help='Path to transcripts JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/guardrails/audio'),
        help='Directory to save generated audio files'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate audio even if files already exist'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Audio File Generation for White-Box Guardrail Evaluation")
    print("=" * 60)
    print(f"Transcripts: {args.transcripts}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Force regenerate: {args.force}")
    print("=" * 60)

    # Check dependencies
    if not GTTS_AVAILABLE:
        print("\nERROR: gTTS not installed")
        print("Install with: pip install gTTS")
        return 1

    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("\nERROR: ffmpeg not found")
        print("Install with: brew install ffmpeg (macOS)")
        return 1

    # Generate audio files
    try:
        stats = generate_all_audio(args.transcripts, args.output_dir, args.force)

        print("\n" + "=" * 60)
        print("Generation Complete")
        print("=" * 60)
        print(f"Success:  {stats['success']}")
        print(f"Skipped:  {stats['skipped']}")
        print(f"Failed:   {stats['failed']}")
        print(f"Total:    {sum(stats.values())}")
        print("=" * 60)

        if stats['failed'] > 0:
            print(f"\n⚠️  {stats['failed']} files failed to generate")
            return 1

        print(f"\n✅ All audio files generated successfully!")
        print(f"Output directory: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
