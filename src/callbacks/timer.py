from datetime import datetime

def log_time(func) -> None:
    def wrapper(self, *args, **kwargs):
        start = datetime.now()

        result = func(self, *args, **kwargs)

        end = datetime.now()

        self.log(f"{func.__name__}_exec_time", (end - start).total_seconds(), prog_bar=True, on_step=True, on_epoch=True)

        return result
    return wrapper