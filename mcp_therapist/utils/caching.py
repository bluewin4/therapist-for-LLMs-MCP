"""
Caching utilities for the MCP Therapist system.

This module provides caching mechanisms to avoid redundant computations
and improve system performance.
"""

import time
import hashlib
import pickle
import functools
import inspect
import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, cast
from threading import RLock
from pathlib import Path

from mcp_therapist.utils.logging import logger
from mcp_therapist.config import settings

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')


class CacheResult:
    """
    Result of a cache operation.
    
    This class encapsulates the result of a cache operation,
    including whether the result was a hit or miss and metadata.
    """
    
    def __init__(
        self,
        value: Any,
        hit: bool,
        time_saved: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cache result.
        
        Args:
            value: The cached value
            hit: Whether the result was a cache hit
            time_saved: Estimated time saved by using the cache
            metadata: Additional metadata about the cache operation
        """
        self.value = value
        self.hit = hit
        self.time_saved = time_saved
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """Get a string representation of the cache result."""
        result_type = "HIT" if self.hit else "MISS"
        return f"Cache{result_type}(time_saved={self.time_saved:.6f}s)"


class MemoryCache:
    """
    In-memory cache implementation.
    
    This class provides a simple in-memory cache with TTL support,
    size limiting, and LRU eviction.
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        ttl: int = 3600,
        enable_stats: bool = True
    ):
        """
        Initialize a memory cache.
        
        Args:
            name: Name of the cache
            max_size: Maximum number of items to store
            ttl: Time-to-live in seconds
            enable_stats: Whether to collect cache statistics
        """
        self.name = name
        self.max_size = max_size
        self.ttl = ttl
        self.enable_stats = enable_stats
        
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = RLock()
        
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._time_saved = 0.0
        
        logger.info(f"MemoryCache '{name}' initialized with max_size={max_size}, ttl={ttl}s")
    
    def get(
        self, 
        key: str,
        computation_time: float = 0.0
    ) -> CacheResult:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            computation_time: Estimated time to compute the value
            
        Returns:
            Cache result
        """
        with self._lock:
            now = time.time()
            
            # Check if key exists and is not expired
            if key in self._cache:
                value, expiry = self._cache[key]
                
                if expiry > now:
                    # Update access time for LRU
                    self._access_times[key] = now
                    
                    # Update stats
                    if self.enable_stats:
                        self._hits += 1
                        self._time_saved += computation_time
                    
                    return CacheResult(
                        value=value,
                        hit=True,
                        time_saved=computation_time,
                        metadata={"expiry": expiry}
                    )
                else:
                    # Expired, remove it
                    del self._cache[key]
                    del self._access_times[key]
            
            # Cache miss
            if self.enable_stats:
                self._misses += 1
            
            return CacheResult(
                value=None,
                hit=False,
                time_saved=0.0
            )
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional override for time-to-live
        """
        with self._lock:
            now = time.time()
            expiry = now + (ttl if ttl is not None else self.ttl)
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            # Store the value
            self._cache[key] = (value, expiry)
            self._access_times[key] = now
            
            # Update stats
            if self.enable_stats:
                self._sets += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info(f"MemoryCache '{self.name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "time_saved": self._time_saved
            }
    
    def _evict_lru(self) -> None:
        """Evict the least recently used item from the cache."""
        if not self._access_times:
            return
        
        # Find the key with the oldest access time
        oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        
        # Remove it
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
        
        # Update stats
        if self.enable_stats:
            self._evictions += 1


class DiskCache:
    """
    Disk-based cache implementation.
    
    This class provides a persistent cache that stores data on disk,
    with TTL support and size limiting.
    """
    
    def __init__(
        self,
        name: str,
        cache_dir: Optional[str] = None,
        max_size_mb: int = 100,
        ttl: int = 86400,
        enable_stats: bool = True
    ):
        """
        Initialize a disk cache.
        
        Args:
            name: Name of the cache
            cache_dir: Directory to store cache files
            max_size_mb: Maximum size in megabytes
            ttl: Time-to-live in seconds
            enable_stats: Whether to collect cache statistics
        """
        self.name = name
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".mcp_therapist", "cache"
        )
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl = ttl
        self.enable_stats = enable_stats
        
        # Create cache directory if it doesn't exist
        self._cache_path = os.path.join(self.cache_dir, name)
        os.makedirs(self._cache_path, exist_ok=True)
        
        # Create metadata file
        self._metadata_path = os.path.join(self._cache_path, "metadata.json")
        self._metadata = self._load_metadata()
        
        self._lock = RLock()
        
        logger.info(
            f"DiskCache '{name}' initialized at {self._cache_path} "
            f"with max_size={max_size_mb}MB, ttl={ttl}s"
        )
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from disk.
        
        Returns:
            Cache metadata
        """
        try:
            if os.path.exists(self._metadata_path):
                with open(self._metadata_path, "r") as f:
                    metadata = json.load(f)
                
                # Update stats from the loaded metadata
                if self.enable_stats:
                    self._hits = metadata.get("stats", {}).get("hits", 0)
                    self._misses = metadata.get("stats", {}).get("misses", 0)
                    self._sets = metadata.get("stats", {}).get("sets", 0)
                    self._evictions = metadata.get("stats", {}).get("evictions", 0)
                    self._time_saved = metadata.get("stats", {}).get("time_saved", 0.0)
                
                return metadata
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
        
        # Initialize stats
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._time_saved = 0.0
        
        # Create default metadata
        return {
            "name": self.name,
            "entries": {},
            "stats": {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "evictions": 0,
                "time_saved": 0.0
            }
        }
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        # Update stats in metadata
        if self.enable_stats:
            self._metadata["stats"] = {
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "time_saved": self._time_saved
            }
        
        try:
            with open(self._metadata_path, "w") as f:
                json.dump(self._metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_path_for_key(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache key
        """
        # Use MD5 for consistent filenames
        hashed = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self._cache_path, f"{hashed}.cache")
    
    def get(
        self, 
        key: str,
        computation_time: float = 0.0
    ) -> CacheResult:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            computation_time: Estimated time to compute the value
            
        Returns:
            Cache result
        """
        with self._lock:
            now = time.time()
            entries = self._metadata.get("entries", {})
            
            # Check if key exists and is not expired
            if key in entries:
                entry = entries[key]
                expiry = entry.get("expiry", 0)
                
                if expiry > now:
                    # Try to load the value from disk
                    try:
                        path = self._get_path_for_key(key)
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                value = pickle.load(f)
                            
                            # Update access time
                            entries[key]["last_access"] = now
                            self._save_metadata()
                            
                            # Update stats
                            if self.enable_stats:
                                self._hits += 1
                                self._time_saved += computation_time
                            
                            return CacheResult(
                                value=value,
                                hit=True,
                                time_saved=computation_time,
                                metadata={"expiry": expiry}
                            )
                    except Exception as e:
                        logger.warning(f"Failed to load cache entry: {e}")
                
                # Either expired or failed to load, remove it
                self._remove_entry(key)
            
            # Cache miss
            if self.enable_stats:
                self._misses += 1
                self._save_metadata()
            
            return CacheResult(
                value=None,
                hit=False,
                time_saved=0.0
            )
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional override for time-to-live
        """
        with self._lock:
            now = time.time()
            expiry = now + (ttl if ttl is not None else self.ttl)
            
            # Check cache size and evict if needed
            self._check_size()
            
            # Store the value on disk
            try:
                path = self._get_path_for_key(key)
                with open(path, "wb") as f:
                    pickle.dump(value, f)
                
                # Update metadata
                size = os.path.getsize(path)
                
                entries = self._metadata.setdefault("entries", {})
                entries[key] = {
                    "expiry": expiry,
                    "size": size,
                    "last_access": now,
                    "created": now
                }
                
                self._save_metadata()
                
                # Update stats
                if self.enable_stats:
                    self._sets += 1
            except Exception as e:
                logger.warning(f"Failed to store cache entry: {e}")
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            entries = self._metadata.get("entries", {})
            
            # Remove all cache files
            for key in list(entries.keys()):
                self._remove_entry(key)
            
            # Reset metadata
            self._metadata = {
                "name": self.name,
                "entries": {},
                "stats": {
                    "hits": 0,
                    "misses": 0,
                    "sets": 0,
                    "evictions": 0,
                    "time_saved": 0.0
                }
            }
            
            self._save_metadata()
            logger.info(f"DiskCache '{self.name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            # Calculate current size
            total_size = 0
            for entry in self._metadata.get("entries", {}).values():
                total_size += entry.get("size", 0)
            
            return {
                "name": self.name,
                "size_bytes": total_size,
                "max_size_bytes": self.max_size_bytes,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "time_saved": self._time_saved,
                "entry_count": len(self._metadata.get("entries", {}))
            }
    
    def _remove_entry(self, key: str) -> bool:
        """
        Remove a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            True if the entry was found and removed, False otherwise
        """
        entries = self._metadata.get("entries", {})
        
        if key in entries:
            # Remove the file
            try:
                path = self._get_path_for_key(key)
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove cache file: {e}")
            
            # Remove from metadata
            del entries[key]
            self._save_metadata()
            return True
        
        return False
    
    def _check_size(self) -> None:
        """
        Check the cache size and evict entries if necessary.
        
        This method removes the least recently accessed entries
        until the cache size is below the maximum.
        """
        entries = self._metadata.get("entries", {})
        
        # Calculate current size
        total_size = 0
        for entry in entries.values():
            total_size += entry.get("size", 0)
        
        # Check if we need to evict
        if total_size > self.max_size_bytes:
            # Sort entries by last access time
            sorted_entries = sorted(
                entries.items(),
                key=lambda item: item[1].get("last_access", 0)
            )
            
            # Remove entries until we're under the limit
            for key, _ in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Add some buffer
                    break
                
                size = entries[key].get("size", 0)
                if self._remove_entry(key):
                    total_size -= size
                    
                    # Update stats
                    if self.enable_stats:
                        self._evictions += 1


# Cache factories
def create_memory_cache(
    name: str,
    max_size: int = 1000,
    ttl: int = 3600
) -> MemoryCache:
    """
    Create a memory cache.
    
    Args:
        name: Name of the cache
        max_size: Maximum number of items to store
        ttl: Time-to-live in seconds
        
    Returns:
        Memory cache instance
    """
    return MemoryCache(name=name, max_size=max_size, ttl=ttl)


def create_disk_cache(
    name: str,
    cache_dir: Optional[str] = None,
    max_size_mb: int = 100,
    ttl: int = 86400
) -> DiskCache:
    """
    Create a disk cache.
    
    Args:
        name: Name of the cache
        cache_dir: Directory to store cache files
        max_size_mb: Maximum size in megabytes
        ttl: Time-to-live in seconds
        
    Returns:
        Disk cache instance
    """
    return DiskCache(name=name, cache_dir=cache_dir, max_size_mb=max_size_mb, ttl=ttl)


# Global cache instances
_embedding_cache = create_memory_cache(
    name="embeddings",
    max_size=getattr(settings, "EMBEDDING_CACHE_SIZE", 1000),
    ttl=getattr(settings, "EMBEDDING_CACHE_TTL", 3600)
)

_resource_cache = create_memory_cache(
    name="mcp_resources",
    max_size=getattr(settings, "RESOURCE_CACHE_SIZE", 500),
    ttl=getattr(settings, "RESOURCE_CACHE_TTL", 1800)
)

_prompt_cache = create_memory_cache(
    name="prompt_templates",
    max_size=getattr(settings, "PROMPT_CACHE_SIZE", 200),
    ttl=getattr(settings, "PROMPT_CACHE_TTL", 3600)
)


# Decorator for memoization with TTL
def memoize(ttl: int = 3600, max_size: int = 1000):
    """
    Decorator for memoizing a function with TTL.
    
    Args:
        ttl: Time-to-live in seconds
        max_size: Maximum cache size
        
    Returns:
        Decorated function
    """
    cache = create_memory_cache(f"memoize_{id(ttl)}_{id(max_size)}", max_size=max_size, ttl=ttl)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function name and arguments
            key_parts = [func.__module__, func.__qualname__]
            
            # Add positional arguments
            for arg in args:
                key_parts.append(str(arg))
            
            # Add keyword arguments (sorted by key)
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            key = ":".join(key_parts)
            
            # Try to get from cache
            start_time = time.time()
            result = cache.get(key, computation_time=0.01)  # Assume 10ms computation time
            
            if result.hit:
                return result.value
            
            # Cache miss, compute the value
            value = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Store in cache
            cache.set(key, value, ttl)
            
            return value
        
        return cast(F, wrapper)
    
    return decorator


# Utility functions for the global caches
def get_embedding_cache() -> MemoryCache:
    """
    Get the global embedding cache.
    
    Returns:
        Embedding cache instance
    """
    return _embedding_cache


def get_resource_cache() -> MemoryCache:
    """
    Get the global MCP resource cache.
    
    Returns:
        Resource cache instance
    """
    return _resource_cache


def get_prompt_cache() -> MemoryCache:
    """
    Get the global prompt template cache.
    
    Returns:
        Prompt cache instance
    """
    return _prompt_cache 