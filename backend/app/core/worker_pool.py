"""
Worker Pool for handling compute-intensive tasks
Optimized for SHAP, LIME, and other ML explainability computations
"""

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class TaskType(Enum):
    """Task types for different processing requirements"""
    CPU_INTENSIVE = "cpu_intensive"  # SHAP, LIME, statistical computations
    IO_INTENSIVE = "io_intensive"    # File operations, API calls
    MEMORY_INTENSIVE = "memory_intensive"  # Large data processing


@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: TaskType
    function: Callable
    args: Tuple
    kwargs: Dict
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None


@dataclass
class TaskResult:
    """Task result"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None


class WorkerPool:
    """
    Advanced worker pool for ML explainability computations
    
    Features:
    - Separate pools for CPU/IO intensive tasks
    - Task prioritization and timeout handling
    - Automatic retry mechanism
    - Resource monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 100,
        cpu_workers: Optional[int] = None,
        io_workers: Optional[int] = None
    ):
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Calculate optimal worker distribution
        self.cpu_workers = cpu_workers or max(1, mp.cpu_count() - 1)
        self.io_workers = io_workers or min(10, max_workers * 2)
        
        # Executors
        self.cpu_executor: Optional[ProcessPoolExecutor] = None
        self.io_executor: Optional[ThreadPoolExecutor] = None
        
        # Task queues
        self.cpu_queue: asyncio.Queue = None
        self.io_queue: asyncio.Queue = None
        
        # Task tracking
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_results: Dict[str, asyncio.Future] = {}
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_tasks": 0,
            "queue_size": 0,
            "cpu_utilization": 0.0,
            "memory_usage": 0.0,
        }
        
        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.processor_tasks: List[asyncio.Task] = []
        
        logger.info(f"WorkerPool initialized: {self.cpu_workers} CPU workers, {self.io_workers} IO workers")
    
    async def start(self):
        """Start the worker pool"""
        if self.running:
            return
        
        self.running = True
        
        # Initialize executors
        self.cpu_executor = ProcessPoolExecutor(
            max_workers=self.cpu_workers,
            mp_context=mp.get_context("spawn")
        )
        self.io_executor = ThreadPoolExecutor(max_workers=self.io_workers)
        
        # Initialize queues
        self.cpu_queue = asyncio.Queue(maxsize=self.queue_size)
        self.io_queue = asyncio.Queue(maxsize=self.queue_size)
        
        # Start task processors
        self.processor_tasks = [
            asyncio.create_task(self._process_cpu_tasks()),
            asyncio.create_task(self._process_io_tasks()),
        ]
        
        # Start monitoring
        self.monitor_task = asyncio.create_task(self._monitor_resources())
        
        logger.info("WorkerPool started successfully")
    
    async def stop(self):
        """Stop the worker pool gracefully"""
        if not self.running:
            return
        
        logger.info("Stopping WorkerPool...")
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for active tasks to complete (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.processor_tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.warning("Some tasks didn't complete within timeout")
        
        # Cancel monitoring
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Shutdown executors
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        
        if self.io_executor:
            self.io_executor.shutdown(wait=True)
        
        logger.info("WorkerPool stopped")
    
    async def submit_task(
        self,
        task_id: str,
        function: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        task_type: TaskType = TaskType.CPU_INTENSIVE,
        timeout: Optional[int] = None,
        callback: Optional[Callable] = None,
        error_callback: Optional[Callable] = None
    ) -> str:
        """Submit a task for execution"""
        if not self.running:
            raise RuntimeError("WorkerPool is not running")
        
        kwargs = kwargs or {}
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            function=function,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            callback=callback,
            error_callback=error_callback
        )
        
        # Create result future
        future = asyncio.Future()
        self.task_results[task_id] = future
        
        # Add to appropriate queue
        if task_type == TaskType.CPU_INTENSIVE or task_type == TaskType.MEMORY_INTENSIVE:
            await self.cpu_queue.put(task)
        else:
            await self.io_queue.put(task)
        
        self.active_tasks[task_id] = task
        self.stats["total_tasks"] += 1
        self.stats["active_tasks"] += 1
        
        logger.debug(f"Task {task_id} submitted ({task_type.value})")
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get task result"""
        if task_id not in self.task_results:
            raise ValueError(f"Task {task_id} not found")
        
        future = self.task_results[task_id]
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"Task {task_id} timed out")
        finally:
            # Cleanup
            self.task_results.pop(task_id, None)
            self.active_tasks.pop(task_id, None)
    
    async def _process_cpu_tasks(self):
        """Process CPU-intensive tasks"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.cpu_queue.get(), timeout=1.0)
                asyncio.create_task(self._execute_task(task, self.cpu_executor))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in CPU task processor: {e}")
    
    async def _process_io_tasks(self):
        """Process IO-intensive tasks"""
        while self.running:
            try:
                task = await asyncio.wait_for(self.io_queue.get(), timeout=1.0)
                asyncio.create_task(self._execute_task(task, self.io_executor))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in IO task processor: {e}")
    
    async def _execute_task(self, task: Task, executor: Union[ProcessPoolExecutor, ThreadPoolExecutor]):
        """Execute a single task"""
        start_time = time.time()
        
        try:
            # Execute task with timeout
            loop = asyncio.get_event_loop()
            
            if task.timeout:
                future = loop.run_in_executor(executor, task.function, *task.args, **task.kwargs)
                result = await asyncio.wait_for(future, timeout=task.timeout)
            else:
                result = await loop.run_in_executor(executor, task.function, *task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            # Create result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                worker_id=f"{executor.__class__.__name__}_{threading.current_thread().name}"
            )
            
            # Call success callback if provided
            if task.callback:
                try:
                    await task.callback(task_result)
                except Exception as e:
                    logger.error(f"Error in task callback: {e}")
            
            # Update statistics
            self.stats["completed_tasks"] += 1
            self.stats["active_tasks"] -= 1
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Task {task.task_id} timed out after {task.timeout}s"
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                
                # Re-submit task
                if task.task_type == TaskType.CPU_INTENSIVE or task.task_type == TaskType.MEMORY_INTENSIVE:
                    await self.cpu_queue.put(task)
                else:
                    await self.io_queue.put(task)
                return
            
            logger.error(error_msg)
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task {task.task_id} failed: {str(e)}"
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
            
            # Call error callback if provided
            if task.error_callback:
                try:
                    await task.error_callback(task_result)
                except Exception as cb_e:
                    logger.error(f"Error in error callback: {cb_e}")
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            self.stats["active_tasks"] -= 1
            
            logger.error(error_msg)
        
        # Set result
        if task.task_id in self.task_results:
            self.task_results[task.task_id].set_result(task_result)
        
        # Store completed task
        self.completed_tasks[task.task_id] = task_result
    
    async def _monitor_resources(self):
        """Monitor resource usage"""
        import psutil
        
        while self.running:
            try:
                # Update statistics
                self.stats["queue_size"] = self.cpu_queue.qsize() + self.io_queue.qsize()
                self.stats["cpu_utilization"] = psutil.cpu_percent(interval=1)
                self.stats["memory_usage"] = psutil.virtual_memory().percent
                
                # Log statistics periodically
                if self.stats["total_tasks"] % 10 == 0:
                    logger.info(f"Worker pool stats: {self.stats}")
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(5)
    
    def status(self) -> Dict[str, Any]:
        """Get worker pool status"""
        return {
            "running": self.running,
            "cpu_workers": self.cpu_workers,
            "io_workers": self.io_workers,
            "stats": self.stats.copy(),
            "queue_sizes": {
                "cpu": self.cpu_queue.qsize() if self.cpu_queue else 0,
                "io": self.io_queue.qsize() if self.io_queue else 0,
            }
        }
    
    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all tasks to complete"""
        start_time = time.time()
        
        while self.stats["active_tasks"] > 0:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Tasks did not complete within timeout")
            
            await asyncio.sleep(0.1)
    
    def get_task_history(self, limit: int = 100) -> List[TaskResult]:
        """Get task execution history"""
        return list(self.completed_tasks.values())[-limit:]