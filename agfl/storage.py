"""Serialize writers to a participant/seed artifact directory."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def run_directory(path):
    """Serialize writers to one seed; process exit releases the OS lock."""
    import fcntl

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Keep the lock file in place: deleting it would let a second process lock
    # a different inode while this process still owns the original lock.
    with (path / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is currently writing this seed: {path}') from error
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


