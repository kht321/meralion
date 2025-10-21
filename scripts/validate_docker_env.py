#!/usr/bin/env python3
"""
Validate that the Docker environment is properly configured.
Run this inside the Docker container to verify all dependencies are installed.
"""

import sys
from pathlib import Path


def check_import(module_name: str, package_name: str = None) -> bool:
    """Try importing a module and report success/failure."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    exists = Path(file_path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {file_path}")
    return exists


def main():
    print("=" * 60)
    print("Docker Environment Validation")
    print("=" * 60)

    all_passed = True

    # Check Python version
    print(f"\n🐍 Python version: {sys.version}")

    # Check core ML packages
    print("\n📦 Core ML Packages:")
    all_passed &= check_import("torch")
    all_passed &= check_import("torchaudio")
    all_passed &= check_import("transformers")
    all_passed &= check_import("datasets")

    # Check audio processing
    print("\n🔊 Audio Processing:")
    all_passed &= check_import("soundfile")
    all_passed &= check_import("librosa")
    all_passed &= check_import("pydub")

    # Check TTS
    print("\n🗣️  Text-to-Speech:")
    all_passed &= check_import("TTS")

    # Check ML utilities
    print("\n🧮 ML Utilities:")
    all_passed &= check_import("numpy")
    all_passed &= check_import("pandas")
    all_passed &= check_import("scipy")
    all_passed &= check_import("sklearn", "scikit-learn")

    # Check visualization
    print("\n📊 Visualization:")
    all_passed &= check_import("matplotlib")
    all_passed &= check_import("seaborn")

    # Check other utilities
    print("\n🛠️  Utilities:")
    all_passed &= check_import("yaml", "pyyaml")
    all_passed &= check_import("jiwer")
    all_passed &= check_import("tqdm")

    # Check project package
    print("\n📁 Project Package:")
    all_passed &= check_import("asr_eval")

    # Check important files
    print("\n📄 Project Files:")
    all_passed &= check_file_exists("requirements.txt")
    all_passed &= check_file_exists("pyproject.toml")
    all_passed &= check_file_exists("asr_eval/__init__.py")

    # Check cache directories
    print("\n📂 Cache Directories:")
    all_passed &= check_file_exists(".cache/torch")
    all_passed &= check_file_exists(".cache/huggingface")

    # Check ffmpeg (system dependency)
    print("\n🎵 System Dependencies:")
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ ffmpeg: {version}")
        else:
            print("❌ ffmpeg: not working")
            all_passed = False
    except Exception as e:
        print(f"❌ ffmpeg: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All checks passed! Environment is ready.")
        return 0
    else:
        print("❌ Some checks failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
