import heapq
import time
import itertools
from collections import deque
from typing import Callable, Any

class MiniScheduler:
    """
    Mini task scheduler using heapq (for scheduled/priority tasks)
    and deque (for ready FIFO execution).
    """

    def __init__(self):
        # heap stores items as (run_at, priority, counter, func, args, kwargs)
        self._heap = []
        self._counter = itertools.count()  # tie-breaker
        self._ready = deque()              # FIFO queue of tasks ready to run
        self._stopped = False

    def schedule_at(self, run_at: float, func: Callable, *args, priority: int = 0, **kwargs):
        """
        Schedule a callable to run at a specific timestamp (epoch seconds).
        Lower priority value means higher priority (0 is default).
        """
        count = next(self._counter)
        item = (run_at, priority, count, func, args, kwargs)
        heapq.heappush(self._heap, item)

    def schedule_after(self, delay_seconds: float, func: Callable, *args, priority: int = 0, **kwargs):
        """Schedule a callable to run after `delay_seconds` from now."""
        run_at = time.time() + delay_seconds
        self.schedule_at(run_at, func, *args, priority=priority, **kwargs)

    def enqueue_immediate(self, func: Callable, *args, urgent: bool = False, **kwargs):
        """
        Put a task directly into the ready queue.
        If urgent=True, place it at the left end so it's executed next.
        """
        if urgent:
            self._ready.appendleft((func, args, kwargs))
        else:
            self._ready.append((func, args, kwargs))

    def _move_due_tasks_to_ready(self):
        """Move all tasks whose run_at <= now from heap to ready deque."""
        now = time.time()
        while self._heap and self._heap[0][0] <= now:
            run_at, priority, count, func, args, kwargs = heapq.heappop(self._heap)
            self._ready.append((func, args, kwargs))

    def run_next(self):
        """Run the next ready task if any. Returns True if a task was run."""
        self._move_due_tasks_to_ready()
        if self._ready:
            func, args, kwargs = self._ready.popleft()
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"[Scheduler] Exception while running task: {e}")
            return True
        return False

    def run_pending(self):
        """Run all ready tasks (move due tasks first)."""
        self._move_due_tasks_to_ready()
        ran_any = False
        while self._ready:
            self.run_next()
            ran_any = True
        return ran_any

    def run_loop(self, poll_interval: float = 0.1, timeout: float = None):
        """
        Run the scheduler loop until there are no scheduled or ready tasks,
        or until timeout seconds have passed (if provided).
        """
        start = time.time()
        while not self._stopped:
            now = time.time()
            if timeout is not None and (now - start) >= timeout:
                break

            # Move due tasks -> ready and run one task (so tasks interleave)
            self._move_due_tasks_to_ready()
            if self._ready:
                self.run_next()
                continue

            # No ready tasks: if no scheduled tasks left, we are done
            if not self._heap:
                break

            # Sleep a bit until next scheduled task or poll_interval
            next_run_at = self._heap[0][0]
            sleep_for = max(0.0, min(poll_interval, next_run_at - time.time()))
            time.sleep(sleep_for)

    def stop(self):
        """Stop the scheduler loop early."""
        self._stopped = True

    def scheduled_count(self):
        return len(self._heap)

    def ready_count(self):
        return len(self._ready)


# -------------------------
# Demo usage
# -------------------------
# if __name__ == "__main__":
#     scheduler = MiniScheduler()

#     # Demo task functions
#     def task(name):
#         print(f"{time.strftime('%H:%M:%S')} - Running task: {name}")

#     def task_with_args(name, x, y):
#         print(f"{time.strftime('%H:%M:%S')} - {name}: {x} + {y} = {x+y}")

#     # 1) enqueue an immediate task (FIFO)
#     scheduler.enqueue_immediate(task, "Immediate-1")

#     # 2) schedule tasks after a delay (uses heapq)
#     scheduler.schedule_after(1.0, task, "Delayed-1 (1s)", priority=1)
#     scheduler.schedule_after(0.5, task_with_args, "Delayed-2 (0.5s)", 5, 7, priority=0)  # higher priority (0)

#     # 3) urgent immediate task (goes to left of deque)
#     scheduler.enqueue_immediate(task, "Urgent-Immediate", urgent=True)

#     # 4) schedule another task later
#     scheduler.schedule_after(2.0, task, "Delayed-3 (2s)", priority=0)

#     print("Starting scheduler loop...\n")
    
#     scheduler.run_loop(poll_interval=0.05)

#     print("\nAll tasks processed.")
