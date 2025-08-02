#!/usr/bin/env python3
"""
Smart Caching System for Ultimate Scanner
Provides persistent hash caching and incremental scanning capabilities
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

class CacheEntry:
    """Represents a cached file entry"""
    
    def __init__(self, path: str, size: int, mtime: float, 
                 md5_hash: Optional[str] = None, 
                 partial_hash: Optional[str] = None,
                 similarity_hashes: Optional[Dict[str, str]] = None,
                 cached_time: Optional[datetime] = None):
        self.path = path
        self.size = size
        self.mtime = mtime
        self.md5_hash = md5_hash
        self.partial_hash = partial_hash
        self.similarity_hashes = similarity_hashes or {}
        self.cached_time = cached_time or datetime.now()
    
    def is_valid(self, current_size: int, current_mtime: float, 
                 mtime_tolerance: float = 1.0) -> bool:
        """Check if cache entry is still valid"""
        # Size must match exactly
        if self.size != current_size:
            return False
        
        # Modification time must be within tolerance (to handle filesystem precision)
        time_diff = abs(self.mtime - current_mtime)
        return time_diff <= mtime_tolerance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'path': self.path,
            'size': self.size,
            'mtime': self.mtime,
            'md5_hash': self.md5_hash,
            'partial_hash': self.partial_hash,
            'similarity_hashes': self.similarity_hashes,
            'cached_time': self.cached_time.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Create from dictionary"""
        cached_time = datetime.fromisoformat(data['cached_time']) if data.get('cached_time') else None
        return cls(
            path=data['path'],
            size=data['size'],
            mtime=data['mtime'],
            md5_hash=data.get('md5_hash'),
            partial_hash=data.get('partial_hash'),
            similarity_hashes=data.get('similarity_hashes', {}),
            cached_time=cached_time
        )

class CacheManager:
    """Manages persistent hash cache with SQLite backend"""
    
    CACHE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS file_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE NOT NULL,
        size INTEGER NOT NULL,
        mtime REAL NOT NULL,
        md5_hash TEXT,
        partial_hash TEXT,
        similarity_hashes TEXT,  -- JSON serialized dict
        cached_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        hit_count INTEGER DEFAULT 0,
        last_hit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_cache_path ON file_cache(path);
    CREATE INDEX IF NOT EXISTS idx_cache_size_mtime ON file_cache(size, mtime);
    CREATE INDEX IF NOT EXISTS idx_cache_md5 ON file_cache(md5_hash);
    CREATE INDEX IF NOT EXISTS idx_cache_partial ON file_cache(partial_hash);
    CREATE INDEX IF NOT EXISTS idx_cache_time ON file_cache(cached_time);
    
    CREATE TABLE IF NOT EXISTS cache_metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE IF NOT EXISTS cache_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_requests INTEGER DEFAULT 0,
        cache_hits INTEGER DEFAULT 0,
        cache_misses INTEGER DEFAULT 0,
        files_cached INTEGER DEFAULT 0,
        bytes_saved INTEGER DEFAULT 0
    );
    """
    
    def __init__(self, cache_path: str = "file_cache.db", 
                 max_cache_size_mb: int = 1000,
                 cleanup_interval_hours: int = 24,
                 mtime_tolerance: float = 1.0):
        """
        Initialize cache manager
        
        Args:
            cache_path: Path to cache database
            max_cache_size_mb: Maximum cache size in MB (0 = unlimited)
            cleanup_interval_hours: How often to run cleanup
            mtime_tolerance: Tolerance for mtime comparison (seconds)
        """
        self.cache_path = cache_path
        self.max_cache_size_mb = max_cache_size_mb
        self.cleanup_interval_hours = cleanup_interval_hours
        self.mtime_tolerance = mtime_tolerance
        
        # Statistics
        self.stats = {
            'requests': 0,
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'bytes_saved': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._last_cleanup = datetime.now()
        
        # Initialize database
        self._init_db()
        self._start_session()
    
    def _init_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            # Performance optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=20000")  # Larger cache for cache DB
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            
            # Create schema
            conn.executescript(self.CACHE_SCHEMA)
            conn.commit()
    
    def _start_session(self):
        """Start a new cache session for statistics"""
        with sqlite3.connect(self.cache_path, timeout=30) as conn:
            conn.execute("""
                INSERT INTO cache_stats (session_start)
                VALUES (?)
            """, (datetime.now(),))
            conn.commit()
    
    def get_cache_entry(self, file_path: str) -> Optional[CacheEntry]:
        """Get cache entry for a file"""
        with self._lock:
            self.stats['requests'] += 1
            
            try:
                with sqlite3.connect(self.cache_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT path, size, mtime, md5_hash, partial_hash, 
                               similarity_hashes, cached_time
                        FROM file_cache
                        WHERE path = ?
                    """, (file_path,))
                    
                    result = cursor.fetchone()
                    if result:
                        path, size, mtime, md5_hash, partial_hash, sim_hashes_json, cached_time = result
                        
                        # Parse similarity hashes
                        try:
                            similarity_hashes = json.loads(sim_hashes_json) if sim_hashes_json else {}
                        except json.JSONDecodeError:
                            similarity_hashes = {}
                        
                        # Parse cached time
                        try:
                            cached_dt = datetime.fromisoformat(cached_time)
                        except:
                            cached_dt = datetime.now()
                        
                        # Update hit statistics
                        cursor.execute("""
                            UPDATE file_cache 
                            SET hit_count = hit_count + 1, last_hit = ?
                            WHERE path = ?
                        """, (datetime.now(), file_path))
                        conn.commit()
                        
                        self.stats['hits'] += 1
                        
                        return CacheEntry(
                            path=path,
                            size=size,
                            mtime=mtime,
                            md5_hash=md5_hash,
                            partial_hash=partial_hash,
                            similarity_hashes=similarity_hashes,
                            cached_time=cached_dt
                        )
                    else:
                        self.stats['misses'] += 1
                        return None
            
            except Exception as e:
                logger.warning(f"Cache lookup failed for {file_path}: {e}")
                self.stats['misses'] += 1
                return None
    
    def store_cache_entry(self, entry: CacheEntry):
        """Store cache entry"""
        with self._lock:
            try:
                with sqlite3.connect(self.cache_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    
                    # Serialize similarity hashes
                    sim_hashes_json = json.dumps(entry.similarity_hashes) if entry.similarity_hashes else None
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO file_cache
                        (path, size, mtime, md5_hash, partial_hash, similarity_hashes, cached_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.path,
                        entry.size,
                        entry.mtime,
                        entry.md5_hash,
                        entry.partial_hash,
                        sim_hashes_json,
                        entry.cached_time.isoformat()
                    ))
                    
                    conn.commit()
                    self.stats['stores'] += 1
                    
                    # Estimate bytes saved
                    if entry.md5_hash:
                        self.stats['bytes_saved'] += entry.size
            
            except Exception as e:
                logger.warning(f"Failed to store cache entry for {entry.path}: {e}")
    
    def invalidate_file(self, file_path: str):
        """Remove file from cache"""
        with self._lock:
            try:
                with sqlite3.connect(self.cache_path, timeout=30) as conn:
                    conn.execute("DELETE FROM file_cache WHERE path = ?", (file_path,))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to invalidate cache for {file_path}: {e}")
    
    def is_file_cached_and_valid(self, file_path: str) -> Tuple[bool, Optional[CacheEntry]]:
        """Check if file is cached and cache entry is valid"""
        try:
            path_obj = Path(file_path)
            if not path_obj.exists():
                return False, None
            
            stat = path_obj.stat()
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            # Get cache entry
            cache_entry = self.get_cache_entry(file_path)
            if not cache_entry:
                return False, None
            
            # Validate cache entry
            is_valid = cache_entry.is_valid(current_size, current_mtime, self.mtime_tolerance)
            return is_valid, cache_entry
        
        except Exception as e:
            logger.debug(f"Cache validation failed for {file_path}: {e}")
            return False, None
    
    def cleanup_cache(self, force: bool = False):
        """Clean up old and invalid cache entries"""
        with self._lock:
            now = datetime.now()
            
            # Check if cleanup is needed
            if not force and (now - self._last_cleanup).total_seconds() < self.cleanup_interval_hours * 3600:
                return
            
            logger.info("Starting cache cleanup...")
            
            try:
                with sqlite3.connect(self.cache_path, timeout=60) as conn:
                    cursor = conn.cursor()
                    
                    # Remove entries for non-existent files
                    cursor.execute("SELECT path FROM file_cache")
                    all_paths = [row[0] for row in cursor.fetchall()]
                    
                    removed_missing = 0
                    removed_invalid = 0
                    
                    for path in all_paths:
                        try:
                            path_obj = Path(path)
                            if not path_obj.exists():
                                cursor.execute("DELETE FROM file_cache WHERE path = ?", (path,))
                                removed_missing += 1
                            else:
                                # Check if entry is still valid
                                stat = path_obj.stat()
                                cursor.execute("""
                                    SELECT size, mtime FROM file_cache WHERE path = ?
                                """, (path,))
                                result = cursor.fetchone()
                                
                                if result:
                                    cached_size, cached_mtime = result
                                    if (cached_size != stat.st_size or 
                                        abs(cached_mtime - stat.st_mtime) > self.mtime_tolerance):
                                        cursor.execute("DELETE FROM file_cache WHERE path = ?", (path,))
                                        removed_invalid += 1
                        
                        except Exception as e:
                            logger.debug(f"Cleanup check failed for {path}: {e}")
                            cursor.execute("DELETE FROM file_cache WHERE path = ?", (path,))
                            removed_invalid += 1
                    
                    # Remove old entries if cache is too large
                    removed_old = 0
                    if self.max_cache_size_mb > 0:
                        # Get cache size
                        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                        cache_size_bytes = cursor.fetchone()[0] if cursor.fetchone() else 0
                        cache_size_mb = cache_size_bytes / (1024 * 1024)
                        
                        if cache_size_mb > self.max_cache_size_mb:
                            # Remove oldest entries (least recently hit)
                            cursor.execute("""
                                DELETE FROM file_cache
                                WHERE id IN (
                                    SELECT id FROM file_cache
                                    ORDER BY last_hit ASC
                                    LIMIT ?
                                )
                            """, (max(100, int(cache_size_mb - self.max_cache_size_mb) * 10),))
                            removed_old = cursor.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"Cache cleanup complete: {removed_missing} missing files, "
                              f"{removed_invalid} invalid entries, {removed_old} old entries removed")
                    
                    self._last_cleanup = now
            
            except Exception as e:
                logger.warning(f"Cache cleanup failed: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            try:
                with sqlite3.connect(self.cache_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    
                    # Get cache size and entry count
                    cursor.execute("SELECT COUNT(*) FROM file_cache")
                    entry_count = cursor.fetchone()[0]
                    
                    cursor.execute("""
                        SELECT 
                            SUM(size) as total_size,
                            AVG(hit_count) as avg_hits,
                            MAX(cached_time) as newest_entry,
                            MIN(cached_time) as oldest_entry
                        FROM file_cache
                    """)
                    result = cursor.fetchone()
                    total_size, avg_hits, newest, oldest = result if result else (0, 0, None, None)
                    
                    # Calculate hit rate
                    total_requests = self.stats['hits'] + self.stats['misses']
                    hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
                    
                    return {
                        'entry_count': entry_count,
                        'total_size_bytes': total_size or 0,
                        'total_size_mb': (total_size or 0) / (1024 * 1024),
                        'avg_hits_per_entry': avg_hits or 0,
                        'newest_entry': newest,
                        'oldest_entry': oldest,
                        'session_stats': self.stats.copy(),
                        'hit_rate_percent': hit_rate,
                        'bytes_saved_mb': self.stats['bytes_saved'] / (1024 * 1024)
                    }
            
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                return {'error': str(e)}
    
    def clear_cache(self):
        """Clear all cache entries"""
        with self._lock:
            try:
                with sqlite3.connect(self.cache_path, timeout=30) as conn:
                    conn.execute("DELETE FROM file_cache")
                    conn.execute("DELETE FROM cache_metadata")
                    conn.execute("DELETE FROM cache_stats")
                    conn.commit()
                
                # Reset statistics
                self.stats = {
                    'requests': 0,
                    'hits': 0,
                    'misses': 0,
                    'stores': 0,
                    'bytes_saved': 0
                }
                
                logger.info("Cache cleared successfully")
            
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
    
    def rebuild_cache(self, scan_paths: List[str]):
        """Rebuild cache by scanning specified paths"""
        self.clear_cache()
        logger.info(f"Rebuilding cache for paths: {scan_paths}")
        
        # This would be called by the main scanner to repopulate the cache
        # Implementation depends on integration with the main scanning logic
        pass
    
    def export_cache(self, export_path: str):
        """Export cache to JSON file"""
        try:
            with sqlite3.connect(self.cache_path, timeout=30) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT path, size, mtime, md5_hash, partial_hash, 
                           similarity_hashes, cached_time, hit_count
                    FROM file_cache
                """)
                
                entries = []
                for row in cursor.fetchall():
                    path, size, mtime, md5, partial, sim_hashes, cached_time, hits = row
                    entries.append({
                        'path': path,
                        'size': size,
                        'mtime': mtime,
                        'md5_hash': md5,
                        'partial_hash': partial,
                        'similarity_hashes': json.loads(sim_hashes) if sim_hashes else {},
                        'cached_time': cached_time,
                        'hit_count': hits
                    })
                
                with open(export_path, 'w') as f:
                    json.dump({
                        'export_time': datetime.now().isoformat(),
                        'entry_count': len(entries),
                        'cache_stats': self.get_cache_stats(),
                        'entries': entries
                    }, f, indent=2)
                
                logger.info(f"Cache exported to {export_path}")
        
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.cleanup_cache(force=True)
        except:
            pass

class IncrementalScanner:
    """Handles incremental scanning using cache"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def should_scan_file(self, file_path: str) -> bool:
        """Determine if file needs to be scanned"""
        is_cached, cache_entry = self.cache.is_file_cached_and_valid(file_path)
        return not is_cached
    
    def get_cached_hashes(self, file_path: str) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
        """Get cached hashes for a file"""
        is_cached, cache_entry = self.cache.is_file_cached_and_valid(file_path)
        
        if is_cached and cache_entry:
            return (cache_entry.md5_hash, 
                   cache_entry.partial_hash, 
                   cache_entry.similarity_hashes)
        
        return None, None, {}
    
    def store_computed_hashes(self, file_path: str, 
                            md5_hash: Optional[str] = None,
                            partial_hash: Optional[str] = None,
                            similarity_hashes: Optional[Dict[str, str]] = None):
        """Store newly computed hashes in cache"""
        try:
            path_obj = Path(file_path)
            stat = path_obj.stat()
            
            entry = CacheEntry(
                path=file_path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                md5_hash=md5_hash,
                partial_hash=partial_hash,
                similarity_hashes=similarity_hashes
            )
            
            self.cache.store_cache_entry(entry)
        
        except Exception as e:
            logger.warning(f"Failed to store computed hashes for {file_path}: {e}")

def get_cache_manager(cache_path: str = "file_cache.db", **kwargs) -> CacheManager:
    """Get a configured cache manager instance"""
    return CacheManager(cache_path, **kwargs)

if __name__ == "__main__":
    # Test cache manager
    print("Testing Cache Manager...")
    
    cache = CacheManager("test_cache.db")
    
    # Test storing and retrieving entries
    test_entry = CacheEntry(
        path="/test/file.txt",
        size=1024,
        mtime=time.time(),
        md5_hash="abc123",
        partial_hash="def456",
        similarity_hashes={"image": "ghi789"}
    )
    
    cache.store_cache_entry(test_entry)
    retrieved = cache.get_cache_entry("/test/file.txt")
    
    print(f"Stored and retrieved entry: {retrieved is not None}")
    print(f"Cache stats: {cache.get_cache_stats()}")
    
    # Cleanup
    os.unlink("test_cache.db")
    print("Cache test completed successfully!")