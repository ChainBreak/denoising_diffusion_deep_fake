
import time

class LoggingScheduler():

    def __init__(self):
        
        self._start_time = self._get_current_time()
        self._last_log_time = self._get_current_time()

        self._last_log_step = None

        self._elapsed_time_since_start = 0.0
        self._elapsed_time_since_last_log = 0.0
        
    def should_we_log_this_step(self, global_step_count):

        if self._is_it_still_the_same_step(global_step_count):
            return True

        if self._has_enough_time_elapsed_since_last_log():
            self._last_log_step = global_step_count
            self._last_log_time = self._get_current_time()
            return True

        return False

    def _is_it_still_the_same_step(self,global_step_count):
        return global_step_count == self._last_log_step
            
    def _has_enough_time_elapsed_since_last_log(self):
        self._update_elapsed_times()

        seconds = 1
        minutes = 60
        hours = 3600

        time_between_logs = 1*hours

        if self._elapsed_time_since_start < 1*minutes:
            time_between_logs = 10*seconds
        elif self._elapsed_time_since_start < 10*minutes:
            time_between_logs = 1*minutes
        elif self._elapsed_time_since_start < 2*hours:
            time_between_logs = 30*minutes

        return self._elapsed_time_since_last_log > time_between_logs

    def _update_elapsed_times(self):
        current_time = self._get_current_time()
        self._elapsed_time_since_start = current_time - self._start_time
        self._elapsed_time_since_last_log = current_time - self._last_log_time

    def _get_current_time(self):
        return time.time()