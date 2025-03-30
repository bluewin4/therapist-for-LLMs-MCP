"""
Performance profiling utilities for the MCP Therapist system.

This module provides tools for identifying bottlenecks and measuring performance
of various system components.
"""

import time
import cProfile
import pstats
import io
import functools
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar
from contextlib import contextmanager

from mcp_therapist.utils.logging import logger

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])


class PerformanceStats:
    """
    Container for performance statistics.
    
    This class tracks execution time, memory usage, and call counts
    for profiled operations.
    """
    
    def __init__(self, name: str):
        """
        Initialize performance stats.
        
        Args:
            name: Name of the profiled component
        """
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.avg_time = 0.0
        self.last_time = 0.0
        self.memory_samples = []
        self.cpu_samples = []
    
    def add_timing(self, elapsed: float) -> None:
        """
        Add a timing measurement.
        
        Args:
            elapsed: Elapsed time in seconds
        """
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.call_count
        self.last_time = elapsed
    
    def add_memory_sample(self, used_bytes: int) -> None:
        """
        Add a memory usage sample.
        
        Args:
            used_bytes: Memory usage in bytes
        """
        self.memory_samples.append(used_bytes)
    
    def add_cpu_sample(self, cpu_percent: float) -> None:
        """
        Add a CPU usage sample.
        
        Args:
            cpu_percent: CPU usage percentage
        """
        self.cpu_samples.append(cpu_percent)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        mem_avg = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_time": self.total_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "last_time": self.last_time,
            "avg_memory_bytes": mem_avg,
            "avg_cpu_percent": cpu_avg
        }
    
    def __str__(self) -> str:
        """Get a string representation of the performance statistics."""
        stats = self.get_summary()
        return (
            f"Performance[{self.name}]: "
            f"calls={stats['call_count']}, "
            f"avg_time={stats['avg_time']:.6f}s, "
            f"min={stats['min_time']:.6f}s, "
            f"max={stats['max_time']:.6f}s, "
            f"mem={stats['avg_memory_bytes']/1024/1024:.2f}MB, "
            f"cpu={stats['avg_cpu_percent']:.1f}%"
        )


class Profiler:
    """
    Performance profiler for MCP Therapist components.
    
    This class provides methods for measuring and analyzing
    performance of system components.
    """
    
    _instance = None
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the profiler."""
        if self._initialized:
            return
        
        self._stats = {}
        self._enabled = False
        self._initialized = True
        self._memory_tracking = False
        logger.info("Performance profiler initialized")
    
    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True
        logger.info("Performance profiling enabled")
    
    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False
        logger.info("Performance profiling disabled")
    
    def enable_memory_tracking(self) -> None:
        """Enable memory tracking."""
        if not self._memory_tracking:
            tracemalloc.start()
            self._memory_tracking = True
            logger.info("Memory tracking enabled")
    
    def disable_memory_tracking(self) -> None:
        """Disable memory tracking."""
        if self._memory_tracking:
            tracemalloc.stop()
            self._memory_tracking = False
            logger.info("Memory tracking disabled")
    
    def get_memory_usage(self) -> int:
        """
        Get current memory usage.
        
        Returns:
            Current memory usage in bytes
        """
        if self._memory_tracking:
            current, peak = tracemalloc.get_traced_memory()
            return current
        return 0
    
    def reset_stats(self) -> None:
        """Reset all performance statistics."""
        self._stats = {}
        logger.info("Performance statistics reset")
    
    def get_stats(self, name: Optional[str] = None) -> Union[Dict[str, PerformanceStats], PerformanceStats, None]:
        """
        Get performance statistics.
        
        Args:
            name: Optional name of the specific component
            
        Returns:
            Performance statistics for the component or all components
        """
        if name:
            return self._stats.get(name)
        return self._stats
    
    def print_stats(self, name: Optional[str] = None) -> None:
        """
        Print performance statistics.
        
        Args:
            name: Optional name of the specific component
        """
        if name and name in self._stats:
            logger.info(str(self._stats[name]))
        else:
            for stat in self._stats.values():
                logger.info(str(stat))
    
    def get_sorted_stats(self, sort_by: str = "avg_time") -> List[Dict[str, Any]]:
        """
        Get performance statistics sorted by a specific metric.
        
        Args:
            sort_by: Metric to sort by
            
        Returns:
            Sorted list of performance statistics
        """
        stats = [s.get_summary() for s in self._stats.values()]
        return sorted(stats, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    def profile_function(self, func: F) -> F:
        """
        Decorator for profiling a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Profiled function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._enabled:
                return func(*args, **kwargs)
            
            name = f"{func.__module__}.{func.__qualname__}"
            if name not in self._stats:
                self._stats[name] = PerformanceStats(name)
            
            # Get initial memory measurement
            if self._memory_tracking:
                tracemalloc.start()
                mem_before = self.get_memory_usage()
            
            # Time the function
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Get final memory measurement
            if self._memory_tracking:
                mem_after = self.get_memory_usage()
                self._stats[name].add_memory_sample(mem_after - mem_before)
                tracemalloc.stop()
            
            # Add timing data
            self._stats[name].add_timing(elapsed)
            
            return result
        
        return wrapper
    
    @contextmanager
    def profile_section(self, name: str):
        """
        Context manager for profiling a section of code.
        
        Args:
            name: Name of the section to profile
            
        Yields:
            None
        """
        if not self._enabled:
            yield
            return
        
        if name not in self._stats:
            self._stats[name] = PerformanceStats(name)
        
        # Get initial memory measurement
        if self._memory_tracking:
            mem_before = self.get_memory_usage()
        
        # Time the section
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            # Get final memory measurement
            if self._memory_tracking:
                mem_after = self.get_memory_usage()
                self._stats[name].add_memory_sample(mem_after - mem_before)
            
            # Add timing data
            self._stats[name].add_timing(elapsed)
    
    def detailed_profile(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """
        Run a detailed profile of a function using cProfile.
        
        Args:
            func: Function to profile
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (function result, profile statistics)
        """
        profiler = cProfile.Profile()
        result = None
        
        try:
            # Start profiling
            profiler.enable()
            
            # Run the function
            result = func(*args, **kwargs)
            
        finally:
            # Stop profiling
            profiler.disable()
            
            # Get statistics
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)  # Print top 20 functions
            stats_str = s.getvalue()
        
        return result, stats_str


# Create a singleton instance
profiler = Profiler()


def profile(func: F) -> F:
    """
    Decorator for profiling a function.
    
    This is a convenience function that calls profiler.profile_function.
    
    Args:
        func: Function to profile
        
    Returns:
        Profiled function
    """
    return profiler.profile_function(func)


@contextmanager
def profile_section(name: str):
    """
    Context manager for profiling a section of code.
    
    This is a convenience function that calls profiler.profile_section.
    
    Args:
        name: Name of the section to profile
        
    Yields:
        None
    """
    with profiler.profile_section(name):
        yield 