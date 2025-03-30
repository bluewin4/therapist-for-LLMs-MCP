"""
Concurrency utilities for the MCP Therapist system.

This module provides utilities for parallel processing and
concurrent execution of tasks.
"""

import asyncio
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast
from functools import wraps
import time
import threading
import logging

from mcp_therapist.utils.logging import logger
from mcp_therapist.config import settings

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')


class TaskManager:
    """
    Manager for concurrent task execution.
    
    This class provides methods for running tasks concurrently,
    with support for limiting, batching, and prioritization.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_async_tasks: Optional[int] = None
    ):
        """
        Initialize the task manager.
        
        Args:
            max_workers: Maximum number of worker threads
            max_async_tasks: Maximum number of concurrent async tasks
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) + 4)
        self.max_async_tasks = max_async_tasks or 10
        
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, 
            thread_name_prefix="mcp_task_"
        )
        self._process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.max_workers // 2)
        )
        
        self._async_semaphore = asyncio.Semaphore(self.max_async_tasks)
        
        # Stats
        self._task_count = 0
        self._completed_count = 0
        self._error_count = 0
        self._total_execution_time = 0.0
        
        # Priority queues
        self._high_priority_queue: List[Tuple[Callable, Tuple, Dict]] = []
        self._normal_priority_queue: List[Tuple[Callable, Tuple, Dict]] = []
        self._low_priority_queue: List[Tuple[Callable, Tuple, Dict]] = []
        
        # Lock for queue access
        self._queue_lock = threading.RLock()
        
        # Event for queue processing
        self._queue_event = threading.Event()
        
        # Queue processor thread
        self._queue_processor = threading.Thread(
            target=self._process_queues,
            daemon=True,
            name="mcp_queue_processor"
        )
        self._queue_processor.start()
        
        # Flag for shutdown
        self._shutdown = False
        
        logger.info(
            f"TaskManager initialized with max_workers={self.max_workers}, "
            f"max_async_tasks={self.max_async_tasks}"
        )
    
    def run_in_thread(
        self, 
        func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> concurrent.futures.Future[T]:
        """
        Run a function in a worker thread.
        
        Args:
            func: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future for the function result
        """
        self._task_count += 1
        future = self._thread_executor.submit(
            self._wrap_task, func, *args, **kwargs
        )
        return future
    
    def run_in_process(
        self, 
        func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> concurrent.futures.Future[T]:
        """
        Run a function in a separate process.
        
        Note: The function must be picklable.
        
        Args:
            func: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Future for the function result
        """
        self._task_count += 1
        future = self._process_executor.submit(
            func, *args, **kwargs
        )
        future.add_done_callback(self._task_completed_callback)
        return future
    
    async def run_async(
        self, 
        coro_func: Callable[..., T], 
        *args, 
        **kwargs
    ) -> T:
        """
        Run a coroutine function with concurrency limiting.
        
        Args:
            coro_func: Coroutine function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the coroutine
        """
        async with self._async_semaphore:
            self._task_count += 1
            start_time = time.time()
            
            try:
                result = await coro_func(*args, **kwargs)
                self._completed_count += 1
                self._total_execution_time += (time.time() - start_time)
                return result
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in async task: {e}")
                raise
    
    def run_in_batch(
        self, 
        func: Callable[[T], R], 
        items: List[T], 
        max_batch_size: int = 10,
        use_processes: bool = False
    ) -> List[R]:
        """
        Run a function on a batch of items concurrently.
        
        Args:
            func: Function to run on each item
            items: List of items to process
            max_batch_size: Maximum batch size
            use_processes: Whether to use processes instead of threads
            
        Returns:
            List of results in the same order as the input items
        """
        if not items:
            return []
        
        # Create batches
        batches = [
            items[i:i+max_batch_size] 
            for i in range(0, len(items), max_batch_size)
        ]
        
        # Process each batch
        results = []
        for batch in batches:
            # Submit all tasks in the batch
            if use_processes:
                futures = [
                    self.run_in_process(func, item) for item in batch
                ]
            else:
                futures = [
                    self.run_in_thread(func, item) for item in batch
                ]
            
            # Wait for all futures to complete
            batch_results = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.ALL_COMPLETED
            )
            
            # Get results in order
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in batch task: {e}")
                    # Add None for failed tasks
                    results.append(None)
        
        return results
    
    async def run_async_batch(
        self, 
        coro_func: Callable[[T], R], 
        items: List[T], 
        max_batch_size: int = 10
    ) -> List[R]:
        """
        Run a coroutine function on a batch of items concurrently.
        
        Args:
            coro_func: Coroutine function to run on each item
            items: List of items to process
            max_batch_size: Maximum batch size
            
        Returns:
            List of results in the same order as the input items
        """
        if not items:
            return []
        
        # Create batches
        batches = [
            items[i:i+max_batch_size] 
            for i in range(0, len(items), max_batch_size)
        ]
        
        # Process each batch
        results = []
        for batch in batches:
            # Create tasks for each item in the batch
            tasks = [
                self.run_async(coro_func, item) for item in batch
            ]
            
            # Wait for all tasks to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in async batch task: {result}")
                    # Add None for failed tasks
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    def queue_task(
        self, 
        func: Callable[..., T], 
        *args,
        priority: str = "normal",
        **kwargs
    ) -> None:
        """
        Queue a task for execution.
        
        Args:
            func: Function to run
            *args: Positional arguments for the function
            priority: Priority level ("high", "normal", or "low")
            **kwargs: Keyword arguments for the function
        """
        with self._queue_lock:
            # Choose the appropriate queue
            if priority == "high":
                queue = self._high_priority_queue
            elif priority == "low":
                queue = self._low_priority_queue
            else:
                queue = self._normal_priority_queue
            
            # Add task to queue
            queue.append((func, args, kwargs))
            
            # Notify queue processor
            self._queue_event.set()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get task execution statistics.
        
        Returns:
            Dictionary of task statistics
        """
        with self._queue_lock:
            high_queue_size = len(self._high_priority_queue)
            normal_queue_size = len(self._normal_priority_queue)
            low_queue_size = len(self._low_priority_queue)
        
        return {
            "task_count": self._task_count,
            "completed_count": self._completed_count,
            "error_count": self._error_count,
            "avg_execution_time": (
                self._total_execution_time / self._completed_count
                if self._completed_count > 0 else 0.0
            ),
            "queue_sizes": {
                "high": high_queue_size,
                "normal": normal_queue_size,
                "low": low_queue_size
            }
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the task manager.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self._shutdown = True
        self._queue_event.set()  # Wake up queue processor
        
        self._thread_executor.shutdown(wait=wait)
        self._process_executor.shutdown(wait=wait)
        
        logger.info("TaskManager shut down")
    
    def _wrap_task(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Wrap a task to track execution time and completion.
        
        Args:
            func: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            self._completed_count += 1
            self._total_execution_time += (time.time() - start_time)
            return result
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in task: {e}")
            raise
    
    def _task_completed_callback(self, future: concurrent.futures.Future) -> None:
        """
        Callback for completed tasks.
        
        Args:
            future: Completed future
        """
        try:
            # Get the result to check for exceptions
            future.result()
            self._completed_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in task: {e}")
    
    def _process_queues(self) -> None:
        """Process tasks from the priority queues."""
        while not self._shutdown:
            # Wait for queue event
            self._queue_event.wait(timeout=1.0)
            self._queue_event.clear()
            
            if self._shutdown:
                break
            
            # Process tasks from queues in priority order
            with self._queue_lock:
                # Check high priority queue
                if self._high_priority_queue:
                    func, args, kwargs = self._high_priority_queue.pop(0)
                    self.run_in_thread(func, *args, **kwargs)
                    continue
                
                # Check normal priority queue
                if self._normal_priority_queue:
                    func, args, kwargs = self._normal_priority_queue.pop(0)
                    self.run_in_thread(func, *args, **kwargs)
                    continue
                
                # Check low priority queue
                if self._low_priority_queue:
                    func, args, kwargs = self._low_priority_queue.pop(0)
                    self.run_in_thread(func, *args, **kwargs)
                    continue


class AsyncTaskGroup:
    """
    Group of async tasks with tracking.
    
    This class provides a convenient way to manage a group of
    async tasks and track their completion.
    """
    
    def __init__(self, limit: Optional[int] = None):
        """
        Initialize an async task group.
        
        Args:
            limit: Maximum number of concurrent tasks
        """
        self.limit = limit
        self.semaphore = asyncio.Semaphore(limit) if limit else None
        self.tasks: Set[asyncio.Task] = set()
    
    async def add(self, coro) -> asyncio.Task:
        """
        Add a coroutine to the task group.
        
        Args:
            coro: Coroutine to add
            
        Returns:
            Created task
        """
        if self.semaphore:
            await self.semaphore.acquire()
        
        task = asyncio.create_task(self._wrap_coro(coro))
        self.tasks.add(task)
        return task
    
    async def _wrap_coro(self, coro):
        """
        Wrap a coroutine to track completion.
        
        Args:
            coro: Coroutine to wrap
            
        Returns:
            Result of the coroutine
        """
        try:
            return await coro
        finally:
            if self.semaphore:
                self.semaphore.release()
    
    async def wait_all(self) -> List[Any]:
        """
        Wait for all tasks to complete.
        
        Returns:
            List of task results
        """
        if not self.tasks:
            return []
        
        # Wait for all tasks to complete
        done, _ = await asyncio.wait(self.tasks)
        
        # Get results
        results = []
        for task in done:
            try:
                results.append(task.result())
            except Exception as e:
                logger.error(f"Error in task group: {e}")
                results.append(None)
        
        # Clear tasks
        self.tasks.clear()
        
        return results
    
    async def __aenter__(self) -> 'AsyncTaskGroup':
        """Enter the async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        await self.wait_all()


# Create a global task manager
task_manager = TaskManager()


def run_in_thread(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Decorator to run a function in a worker thread.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> concurrent.futures.Future[T]:
        return task_manager.run_in_thread(func, *args, **kwargs)
    return wrapper


def run_in_process(func: Callable[..., T]) -> Callable[..., concurrent.futures.Future[T]]:
    """
    Decorator to run a function in a separate process.
    
    Note: The function must be picklable.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> concurrent.futures.Future[T]:
        return task_manager.run_in_process(func, *args, **kwargs)
    return wrapper


def async_limiter(limit: int = 10):
    """
    Decorator to limit concurrent executions of an async function.
    
    Args:
        limit: Maximum number of concurrent executions
        
    Returns:
        Decorator function
    """
    # Create a semaphore for the function
    semaphore = asyncio.Semaphore(limit)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    
    return decorator


async def gather_with_concurrency(limit: int, *coros):
    """
    Run coroutines concurrently with a limit.
    
    Args:
        limit: Maximum number of concurrent coroutines
        *coros: Coroutines to run
        
    Returns:
        List of results
    """
    async with AsyncTaskGroup(limit) as group:
        for coro in coros:
            await group.add(coro)
        
        return await group.wait_all()


def shutdown() -> None:
    """Shut down the global task manager."""
    task_manager.shutdown()


# Clean up on module unload
import atexit
atexit.register(shutdown) 