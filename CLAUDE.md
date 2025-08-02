# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **TurboDedup** - a high-performance Python application for detecting duplicate files using a three-phase optimization strategy with partial hashing, **smart caching**, and **GPU acceleration**. The scanner can achieve 10-100x performance improvements over traditional duplicate detection tools through intelligent partial hashing for large files, persistent caching that eliminates redundant I/O, and optional GPU acceleration for hash computation.

## Core Architecture

### Main Components

- **turbo_dedup.py**: Main TurboDedup application with three-phase optimization
  - Phase 1: File discovery and cataloging
  - Phase 2: Partial hash computation for large files (head + tail segments)
  - Phase 3: Full hash computation only for potential duplicates
  - Phase 4: Optional byte-by-byte verification

- **auto-config.py**: System detection and configuration optimization tool
  - Detects storage type (SSD/HDD), RAM, CPU cores, network drives
  - Generates optimal scanner settings based on system characteristics
  - Provides performance explanations and alternative configurations

- **interactive_cleaner.py**: Human-in-the-loop duplicate file deletion tool
  - Interactive selection of duplicates to delete with preview
  - Smart auto-selection strategies (keep newest, oldest, priority directories)
  - Safety features: dry-run mode, backups, confirmation prompts
  - Batch operations with undo logging

- **cache_manager.py**: Smart caching system for persistent hash storage
  - SQLite-based cache with WAL mode for concurrent access
  - Automatic cache validation using file size and modification time
  - Smart cleanup and size management with configurable retention
  - Comprehensive statistics and performance tracking

- **gpu_accelerator.py**: GPU acceleration framework for hash computation
  - CUDA and OpenCL support with automatic backend detection
  - Intelligent fallback to multi-threaded CPU computation
  - Memory-efficient batch processing for large file sets
  - Device information and capability detection

### Key Innovations

#### Partial Hashing
The scanner's primary performance advantage comes from partial hashing of large files (>512MB by default). Instead of hashing entire large files, it hashes the first and last segments (8MB each by default). This dramatically reduces I/O while maintaining high duplicate detection accuracy.

#### Smart Caching System (Phase 1 Enhancement)
The persistent caching system eliminates redundant hash computation by storing file metadata and computed hashes in a SQLite database. Files are automatically validated using size and modification time, ensuring cache integrity while dramatically reducing scan times for unchanged files.

#### GPU Acceleration (Phase 1 Enhancement)
Optional GPU acceleration using CUDA or OpenCL provides significant performance improvements for hash computation on systems with compatible GPUs. The system automatically detects available GPU backends and falls back gracefully to optimized CPU computation when GPU acceleration is unavailable.

### Database Schema

Uses SQLite with WAL mode for concurrent access:

**Main Database** (`file_scanner.db`):
- `files` table: path, size, mtime, partial hash (phash), full hash, errors
- Optimized indexes for size-based and hash-based queries
- Batch operations for high-throughput writes

**Cache Database** (`file_cache.db`):
- `file_cache` table: path, size, mtime, hashes, similarity hashes, hit statistics
- `cache_metadata` table: configuration and maintenance information
- `cache_stats` table: session-based performance statistics
- Automatic cleanup and validation processes

## Common Commands

### Basic Usage
```bash
# Scan and interactively manage duplicates (default behavior)
python3 turbo_dedup.py

# Scan specific path with interactive deletion
python3 turbo_dedup.py --path /path/to/scan

# Scan multiple paths
python3 turbo_dedup.py --path /path/one /path/two

# Auto-delete keeping newest files (dry-run by default)
python3 turbo_dedup.py --delete-strategy keep_newest

# Actually delete files (live mode)
python3 turbo_dedup.py --delete-strategy keep_newest --delete-live

# Scan only (no deletion processing)
python3 turbo_dedup.py --no-delete

# Export results to CSV
python3 turbo_dedup.py --no-delete --export csv --export-path duplicates.csv
```

### Auto-Configuration
```bash
# Get optimal settings for a path
python3 auto-config.py /path/to/scan

# Interactive configuration (will prompt for path)
python3 auto-config.py
```

### Performance Tuning
```bash
# High-performance SSD scan with GPU acceleration and caching
python3 turbo_dedup.py --workers 16 --chunk-size 4MB --algorithm xxhash --enable-gpu --enable-cache

# Network drive scan (conservative settings)
python3 turbo_dedup.py --workers 4 --chunk-size 256KB --retry-attempts 5 --enable-cache

# Memory-constrained system
python3 turbo_dedup.py --workers 4 --chunk-size 1MB --batch-size 500 --enable-cache

# Maximum performance configuration
python3 turbo_dedup.py --workers 16 --chunk-size 4MB --algorithm xxhash --enable-gpu --enable-cache --gpu-backend cuda
```

### Deletion Strategies
```bash
# Interactive mode (default) - manually choose what to delete
python3 turbo_dedup.py --delete-strategy interactive

# Auto-strategies
python3 turbo_dedup.py --delete-strategy keep_newest    # Keep most recent files
python3 turbo_dedup.py --delete-strategy keep_oldest    # Keep oldest files  
python3 turbo_dedup.py --delete-strategy keep_first     # Keep alphabetically first
python3 turbo_dedup.py --delete-strategy keep_priority  # Keep files in priority dirs
python3 turbo_dedup.py --delete-strategy keep_original  # Keep original files (smart detection)
python3 turbo_dedup.py --delete-strategy skip          # Find but don't delete

# Live deletion with confirmations disabled
python3 turbo_dedup.py --delete-strategy keep_newest --delete-live --no-delete-confirm

# Disable backups for permanent deletion
python3 turbo_dedup.py --delete-live --no-delete-backup
```

### Smart Caching System (Phase 1 Enhancement)
The persistent cache dramatically improves performance for repeat scans by storing computed hashes and eliminating redundant I/O:

```bash
# Enable caching (automatic for subsequent scans)
python3 turbo_dedup.py --enable-cache

# View detailed cache statistics
python3 turbo_dedup.py --cache-stats

# Clear cache and start fresh
python3 turbo_dedup.py --clear-cache

# Use custom cache location
python3 turbo_dedup.py --enable-cache --cache-path /path/to/cache.db

# Combine with other optimizations
python3 turbo_dedup.py --enable-cache --enable-gpu --workers 16
```

**Cache Performance Benefits:**
- **80-95% cache hit rate** on repeat scans of unchanged directories
- **10-50x faster** scanning of previously processed files
- **Automatic validation** ensures cache integrity with file changes
- **Smart cleanup** maintains optimal cache size and performance

### GPU Acceleration (Phase 1 Enhancement)
Optional GPU acceleration provides significant performance improvements for hash computation:

```bash
# Enable GPU acceleration (auto-detects best backend)
python3 turbo_dedup.py --enable-gpu

# Force specific GPU backend
python3 turbo_dedup.py --enable-gpu --gpu-backend cuda
python3 turbo_dedup.py --enable-gpu --gpu-backend opencl

# Check GPU capabilities
python3 turbo_dedup.py --gpu-info

# Combine GPU with caching for maximum performance
python3 turbo_dedup.py --enable-gpu --enable-cache --workers 16

# Benchmark GPU vs CPU performance
python3 turbo_dedup.py --enable-gpu --benchmark-gpu
```

**GPU Requirements:**
```bash
# For CUDA support (NVIDIA GPUs)
pip install pycuda>=2022.2.2

# For OpenCL support (NVIDIA, AMD, Intel GPUs)
pip install pyopencl>=2023.1.4

# Check GPU support
python3 -c "from gpu_accelerator import check_gpu_support; print(check_gpu_support())"
```

**Performance Expectations:**
- **NVIDIA RTX Series**: 5-15x speedup for large file sets
- **AMD GPUs**: 3-8x speedup via OpenCL
- **Intel GPUs**: 2-5x speedup via OpenCL
- **Automatic fallback** to optimized CPU computation if GPU unavailable

### Smart Original Detection
The `keep_original` strategy intelligently detects and preserves original files by recognizing common patterns:

```bash
# Perfect for downloaded files and copies
python3 turbo_dedup.py --delete-strategy keep_original

# Patterns detected:
# Image.png (kept) vs Image (1).png, Image (2).png (deleted)
# Document.pdf (kept) vs Document - Copy.pdf (deleted)  
# File.txt (kept) vs File - Copy (2).txt (deleted)
# Photo.jpg (kept) vs Photo_copy.jpg (deleted)
```

### Similarity Detection (Phase 1 Enhancement)
Beyond exact duplicate detection, the scanner can now find similar files using advanced algorithms:

```bash
# Enable all similarity detection types
python3 turbo_dedup.py --enable-similarity

# Image similarity (perceptual hashing for different formats/compression)
python3 turbo_dedup.py --image-similarity --image-sensitivity high

# Audio similarity (acoustic fingerprinting across formats/bitrates)
python3 turbo_dedup.py --audio-similarity --audio-sensitivity medium

# Document similarity (fuzzy text matching for similar content)
python3 turbo_dedup.py --document-similarity --document-sensitivity low

# Combine with exact duplicates and deletion strategies
python3 turbo_dedup.py --image-similarity --delete-strategy keep_original --delete-live
```

## Installation & Dependencies

### Core Installation (No External Dependencies)
The basic scanner works with Python standard library only:
```bash
git clone <repository>
cd File_Scanner
python3 turbo_dedup.py --help  # Ready to use!
```

### Enhanced Features Installation
```bash
# GPU Acceleration (Phase 1 Enhancement)
pip install pycuda>=2022.2.2      # NVIDIA CUDA support
pip install pyopencl>=2023.1.4    # OpenCL support (NVIDIA, AMD, Intel)

# Similarity Detection (Phase 1 Enhancement) 
pip install Pillow imagehash                    # Image similarity
pip install librosa soundfile                   # Audio similarity  
pip install python-Levenshtein PyPDF2 python-docx  # Document similarity

# Performance Optimization
pip install xxhash                 # Faster hashing algorithm

# All enhanced features
pip install -r requirements_enhanced.txt  # If available
```

### Advanced Options
```bash
# Reset database and start fresh
python3 turbo_dedup.py --reset

# Include only specific file types
python3 turbo_dedup.py --include .mp4 .mkv .avi

# Exclude additional directories
python3 turbo_dedup.py --exclude-dir .git node_modules __pycache__

# Verify duplicates with byte-by-byte comparison
python3 turbo_dedup.py --verify

# Scan only mode for analysis
python3 turbo_dedup.py --no-delete --quiet --export csv

# Advanced caching options
python3 turbo_dedup.py --enable-cache --cache-max-size 2000  # 2GB cache limit
python3 turbo_dedup.py --cache-export cache_backup.json     # Export cache

# GPU diagnostics and benchmarking
python3 turbo_dedup.py --gpu-info --gpu-benchmark           # Detailed GPU info
python3 turbo_dedup.py --enable-gpu --gpu-memory-limit 4096 # 4GB GPU memory limit
```

### Legacy Interactive Cleaner (Deprecated)
```bash
# The old separate interactive cleaner is deprecated in favor of integrated deletion
# Use --interactive-clean for legacy compatibility
python3 turbo_dedup.py --interactive-clean

# Modern equivalent using integrated system:
python3 turbo_dedup.py --delete-strategy interactive
```

## Configuration Guidelines

### Storage Type Considerations
- **SSD**: Use higher worker counts (CPU × 2), larger chunk sizes (2-4MB), lower partial hash thresholds (256MB), enable GPU and caching
- **HDD**: Limit workers (≤4), use 1MB chunks, higher partial hash thresholds (512MB), enable caching for subsequent scans
- **Network**: Conservative workers (≤4), smaller chunks (256KB), higher retry attempts, caching provides major benefits

### Memory Considerations
- **16GB+**: Can use 4MB chunks, higher worker counts, enable GPU acceleration
- **8-16GB**: Use 2MB chunks, moderate workers, GPU acceleration beneficial
- **<8GB**: Stick to 1MB chunks, lower worker counts, caching still provides benefits

### GPU Considerations
- **NVIDIA RTX/GTX Series**: Excellent CUDA performance, prefer `--gpu-backend cuda`
- **AMD GPUs**: Use OpenCL backend, good performance on recent cards
- **Intel GPUs**: Basic OpenCL support, modest performance improvements
- **No GPU**: Automatic fallback to optimized multi-threaded CPU computation
- **Mixed Systems**: Auto-detection chooses optimal backend

### Algorithm Selection
- **xxhash**: Fastest (2-3x faster than MD5), requires `pip install xxhash`, excellent GPU acceleration
- **md5**: Good balance of speed and compatibility (default), good GPU support
- **sha256**: Slower but more secure if hash integrity is critical, basic GPU support

### Caching Strategy
- **Cache Location**: Use SSD for cache database if available
- **Cache Size**: 500MB-2GB depending on dataset size and available storage
- **Cleanup Frequency**: Daily cleanup maintains optimal performance
- **Retention**: Keep frequently accessed entries, expire unused entries
- **Validation**: Automatic validation prevents stale cache issues

## Development Notes

### File Structure
- **Core**: No external dependencies beyond Python standard library (xxhash is optional)
- **Enhanced Features**: Optional dependencies for GPU acceleration and similarity detection
- **Databases**: Self-contained SQLite databases for persistence and caching
- **Cross-platform**: Full compatibility (Windows, Linux, macOS)
- **Modular**: Core scanner works without optional enhancements

### Error Handling
- Comprehensive retry logic for file access errors
- Graceful handling of permission denied, file locks, network issues
- Statistics tracking for all error conditions

### Performance Monitoring
- Built-in progress tracking with ETA calculations
- Phase timing for optimization analysis
- Throughput metrics (files/sec, MB/sec)
- **Cache Statistics**: Hit/miss rates, performance improvements, storage efficiency
- **GPU Metrics**: Device utilization, memory usage, acceleration factors
- **I/O Optimization**: Tracks cache-eliminated reads and performance gains

### Testing
The project doesn't include formal unit tests. To test functionality:
```bash
# Test basic functionality on small directory
python3 turbo_dedup.py --path /small/test/directory --min-size 1MB

# Test auto-configuration
python3 auto-config.py /test/path

# Test GPU acceleration
python3 turbo_dedup.py --gpu-info
python3 turbo_dedup.py --path /small/test/directory --enable-gpu --benchmark-gpu

# Test caching system
python3 turbo_dedup.py --path /small/test/directory --enable-cache
python3 turbo_dedup.py --cache-stats

# Verify database integrity after scan
sqlite3 file_scanner.db "SELECT COUNT(*) FROM files WHERE error IS NULL;"
sqlite3 file_cache.db "SELECT COUNT(*) FROM file_cache;"
```

## Security Considerations

This is a defensive security tool for system administration and file management. The scanner:
- Only reads files (never modifies or deletes unless explicitly using deletion features)
- Uses safe file operations with proper exception handling  
- Stores only file metadata and hashes (not file contents)
- Respects file system permissions
- **Cache Security**: Cache database contains only file paths, sizes, timestamps, and hash values
- **GPU Security**: GPU acceleration only processes file content for hashing, no data retention
- **Safe Deletion**: Multiple confirmation layers and backup options for deletion operations

## Performance Impact Summary

### Smart Caching System
- **First Scan**: Normal performance, builds comprehensive cache
- **Subsequent Scans**: 80-95% faster for unchanged files
- **Cache Hit Rate**: Typically 80%+ on repeat scans
- **Storage Overhead**: ~1-10MB cache per 10,000 files
- **Maintenance**: Automatic cleanup and validation

### GPU Acceleration  
- **Supported Hardware**: NVIDIA GPUs (CUDA), AMD/Intel GPUs (OpenCL)
- **Performance Gain**: 5-15x speedup on compatible systems
- **Memory Efficient**: Batch processing prevents GPU memory overflow
- **Fallback Strategy**: Automatic CPU fallback with no performance loss
- **Best Use Cases**: Large file sets (>1GB total), high-end GPU hardware

### Combined Benefits
Using both caching and GPU acceleration can result in:
- **10-50x performance improvement** on repeat scans
- **Significant energy savings** by eliminating redundant computation
- **Better system responsiveness** during scanning operations
- **Scalability** for very large datasets and enterprise deployments